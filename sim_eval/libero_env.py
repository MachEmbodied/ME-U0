"""LIBERO simulation environment for evaluation.

Wraps the LIBERO benchmark (libero_spatial, libero_object, libero_goal,
libero_10, libero_90) behind the :class:`BaseSimEnv` interface.

This file runs inside the **conda libero env** — no torch / no leap package
imports allowed. It only talks to a remote ``leap serve`` process over
:class:`sim_eval.client.PolicyClient` (msgpack + websocket).

Training↔eval alignment (following starVLA's ``eval_libero.py``):

* State (8-D): ``[eef_pos(3), axis_angle(quat)(3), gripper_qpos(2)]`` — matches
  what the LeRobot parquet ``observation.state`` column stores.
* Image: ``obs["agentview_image"]`` and ``obs["robot0_eye_in_hand_image"]``,
  each rotated 180° (``[::-1, ::-1]``), resized to ``image_size``.
* Action chunk: the server returns ``T`` future actions in one shot; this env
  caches the chunk and only calls the server every ``action_chunk_size`` steps.
* Success detection: uses ``done`` flag from ``env.step()`` (``info["success"]``
  is not reliable in LIBERO 1.0).

Requires: LIBERO installation with ``OffScreenRenderEnv`` that accepts the
``bddl_file_name`` kwarg (vla-benchmark LIBERO).
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Optional, Tuple

import numpy as np

from sim_eval.base_env import BaseSimEnv
from sim_eval.utils import quat2axisangle, resize_image

logger = logging.getLogger(__name__)


def _ensure_libero_config() -> None:
    """Pre-create ``$LIBERO_CONFIG_PATH/config.yaml`` with defaults so
    LIBERO's ``__init__.py`` doesn't ``input()``-prompt a non-tty worker,
    **and** touch the referenced folders so LIBERO's
    ``get_libero_path()`` stops re-printing ``[Warning]: datasets path ...``
    on every call.

    LIBERO 1.0's top-level ``libero/libero/__init__.py`` reads
    ``$LIBERO_CONFIG_PATH/config.yaml`` on import; when the file doesn't
    exist it runs ``input("Do you want to specify a custom path ...")`` —
    which in a subprocess with no stdin raises ``EOFError`` and aborts
    the whole import chain. The real error disappears behind misleading
    ``MjRenderContext.__del__`` messages from the GC's desperate cleanup
    of partially-built renderers.

    Idempotent: if the config already exists we don't overwrite it, but
    we still ``mkdir -p`` any referenced directories that are missing
    (the unused ``datasets/`` especially — LIBERO doesn't need data at
    eval time but keeps complaining it's absent).
    """
    libero_config_path = os.environ.get(
        "LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")
    )
    config_file = os.path.join(libero_config_path, "config.yaml")

    # Use LIBERO's own default paths. We hard-code the layout to avoid
    # importing ``libero.libero`` (which would trigger the interactive prompt).
    libero_home = os.environ.get("LIBERO_HOME")
    if libero_home:
        benchmark_root = os.path.join(libero_home, "libero", "libero")
    else:
        try:
            import importlib.util
            spec = importlib.util.find_spec("libero.libero")
            if spec is None or spec.origin is None:
                return
            benchmark_root = os.path.dirname(spec.origin)
        except Exception:
            return

    defaults = {
        "benchmark_root": benchmark_root,
        "bddl_files": os.path.join(benchmark_root, "./bddl_files"),
        "init_states": os.path.join(benchmark_root, "./init_files"),
        "datasets": os.path.join(benchmark_root, "../datasets"),
        "assets": os.path.join(benchmark_root, "./assets"),
    }

    # 1) make sure every referenced dir exists (silences the warning)
    for path in defaults.values():
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass

    # 2) create the config file if missing (skips LIBERO's interactive prompt)
    if not os.path.exists(config_file):
        try:
            import yaml
            os.makedirs(libero_config_path, exist_ok=True)
            with open(config_file, "w") as f:
                yaml.safe_dump(defaults, f)
            logger.info("Created LIBERO config at %s", config_file)
        except Exception as e:
            logger.warning("Could not pre-create LIBERO config.yaml: %s", e)


# Max episode steps per task suite — matches starVLA / FluxVLA defaults.
_MAX_STEPS_MAP = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

# First few dummy actions let the simulator settle the objects.
LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


class LiberoEnv(BaseSimEnv):
    """LIBERO benchmark evaluation env, aligned with LeRobot training data."""

    num_steps_wait = 10
    dummy_action = LIBERO_DUMMY_ACTION

    def __init__(
        self,
        task_suite: str = "libero_10",
        resolution: int = 256,
        image_size: int = 224,
        action_chunk_size: Optional[int] = None,
        normalizer_source: Optional[str] = None,
    ) -> None:
        """
        Args:
            task_suite: One of libero_spatial / libero_object / libero_goal /
                libero_10 / libero_90.
            resolution: Render resolution requested from ``OffScreenRenderEnv``
                (before resize). starVLA uses 256.
            image_size: Final image size sent to the server (matches training
                ``image_size``). Usually 224.
            action_chunk_size: Number of future actions consumed per server
                call. ``None`` (default) means *use the full chunk the server
                returns*, which aligns with training ``action_horizon``
                (16 for libero_goal). Smaller values trade latency for
                responsiveness but effectively throw away predictions.
            normalizer_source: Optional training source key used only for
                server-side normalizer routing. Keep ``None`` for original
                LIBERO; pass ``libero_plus`` for LIBERO-plus checkpoints.
        """
        self.task_suite = task_suite
        self.resolution = resolution
        self.image_size = image_size
        self.action_chunk_size = action_chunk_size  # None → use full chunk
        self.normalizer_source = normalizer_source

        # Lazy-loaded benchmark objects
        self._benchmark = None
        self._task_descriptions = None

    # ------------------------------------------------------------------
    # Pickling support for multiprocessing spawn — drop LIBERO class refs
    # so each worker re-imports them under its own sys.path entries.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_benchmark"] = None
        state["_task_descriptions"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._benchmark = None
        self._task_descriptions = None

    @property
    def max_steps(self) -> int:
        return _MAX_STEPS_MAP.get(self.task_suite, 400)

    # ------------------------------------------------------------------
    # Benchmark / env construction
    # ------------------------------------------------------------------

    def _get_benchmark(self):
        """Lazily load the LIBERO benchmark.

        LIBERO emits two cosmetic ``print`` lines we'd rather not see N times
        (one per ``spawn`` worker):

        * ``[Warning]: datasets path ... does not exist!`` — at module import
          time, because the unused ``datasets/`` folder is absent from the
          vla-benchmark checkout.
        * ``[info] using task orders [0, 1, 2, …, 9]`` — at benchmark
          construction time.

        Both go through plain ``print`` (not Python logging), so we
        wrap the whole ``import + instantiate`` block in a
        :func:`contextlib.redirect_stdout` to a ``StringIO`` sink.

        Disable suppression for debugging via ``LIBERO_SUPPRESS_INIT_WARN=0``.
        """
        if self._benchmark is None:
            import contextlib
            import io
            import os
            import sys

            _ensure_libero_config()

            suppress_warn = os.environ.get("LIBERO_SUPPRESS_INIT_WARN", "1") != "0"
            sink = io.StringIO() if suppress_warn else sys.stdout
            # redirect both stdout AND stderr in case some versions of
            # LIBERO emit to stderr.
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                from libero.libero import benchmark as bm

                bench_cls = bm.get_benchmark(self.task_suite)
                self._benchmark = bench_cls()
                self._task_descriptions = [
                    self._benchmark.get_task(i).language
                    for i in range(self._benchmark.n_tasks)
                ]
        return self._benchmark

    def task_count(self) -> int:
        return self._get_benchmark().n_tasks

    def task_descriptions(self):
        self._get_benchmark()
        return self._task_descriptions

    def make_env(self, task_id: int, seed: int) -> Tuple[Any, str]:
        """Create a LIBERO ``OffScreenRenderEnv`` for one task.

        The first-time import of ``libero.libero.envs`` transitively pulls
        ``robosuite`` and ``gym``, both of which shout deprecation banners
        to stdout / stderr on import. Silence those around the imports;
        disable with ``LIBERO_SUPPRESS_INIT_WARN=0`` to debug.
        """
        import contextlib
        import io
        import os
        import sys

        _ensure_libero_config()

        suppress_warn = os.environ.get("LIBERO_SUPPRESS_INIT_WARN", "1") != "0"
        sink = io.StringIO() if suppress_warn else sys.stdout
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            from libero.libero import get_libero_path
            from libero.libero.envs import OffScreenRenderEnv

            benchmark = self._get_benchmark()
            task = benchmark.get_task(task_id)

            # vla-benchmark LIBERO's OffScreenRenderEnv requires ``bddl_file_name``
            # (not ``bddl_file``). The file lives under the configured bddl_files
            # root / the task's problem_folder / bddl file name — matches starVLA.
            #
            # NOTE: ``get_libero_path`` re-emits ``[Warning]: datasets path ...``
            # on EVERY call (not just at import time), so it must stay inside
            # the redirect — hence the whole block is in one ``with``.
            task_bddl_file = (
                pathlib.Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            )

            # Construction of OffScreenRenderEnv also emits robosuite chatter
            # (texture loading info etc.); keep it suppressed too.
            env = OffScreenRenderEnv(
                bddl_file_name=str(task_bddl_file),
                camera_heights=self.resolution,
                camera_widths=self.resolution,
            )
            env.seed(seed)
        return env, self._task_descriptions[task_id]

    def reset_env(self, env: Any, task_id: int, episode_idx: int) -> Any:
        """Reset the env and apply this episode's initial state.

        LIBERO stores per-task ``init_states`` (typically 50 per task).
        Using them is the ONLY way to get diverse per-episode start
        configurations — plain ``env.reset()`` always produces the same
        deterministic spawn, which is why without this path every video
        looks identical. Matches starVLA's ``eval_libero.py`` exactly:

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

        If the benchmark has fewer init_states than episodes (can happen
        with ``--episodes > 50``) we wrap with modulo so every episode
        still gets SOME init_state, just potentially repeated.

        We log (WARNING) — not silently swallow — when init_states are
        missing: that's a data-quality bug, not a normal operating mode.
        """
        import contextlib
        import io
        import os
        import sys

        benchmark = self._get_benchmark()

        # LIBERO's `get_task_init_states` calls `get_libero_path("init_states")`
        # which re-prints the datasets-missing warning; wrap in redirect.
        suppress_warn = os.environ.get("LIBERO_SUPPRESS_INIT_WARN", "1") != "0"
        sink = io.StringIO() if suppress_warn else sys.stdout

        env.reset()
        init_states = None
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                init_states = benchmark.get_task_init_states(task_id)
        except Exception as exc:
            logger.warning(
                "reset_env: get_task_init_states(task=%d) raised: %s — "
                "falling back to plain env.reset() (scene will NOT vary per episode).",
                task_id, exc,
            )

        if init_states is None or len(init_states) == 0:
            logger.warning(
                "reset_env: no init_states found for task=%d; all episodes will start "
                "from the same spawn state. Check that $LIBERO_HOME points at the "
                "vla-benchmark LIBERO checkout and that init_files/ is populated.",
                task_id,
            )
            return env.reset()

        # Modulo wrap in case episodes > len(init_states). Most LIBERO
        # task suites ship exactly 50 init_states, so --episodes=50 is
        # the "one init_state per episode" sweet spot.
        idx = int(episode_idx) % len(init_states)
        return env.set_init_state(init_states[idx])

    # ------------------------------------------------------------------
    # Observation / action conversion — matches starVLA's eval_libero.py
    # ------------------------------------------------------------------

    def get_observation(
        self, env: Any, raw_obs: Any, task_description: str
    ) -> dict:
        """Turn a raw LIBERO obs into the server request payload.

        Server receives:
            - ``images``: list of ``uint8 (H, W, 3)`` — main + wrist cams
              (180° rotated, resized to ``image_size``).
            - ``state``: ``float32 (8,)`` — 3 pos + 3 axis-angle + 2 gripper.
            - ``instruction``: str.
        """
        # 180° rotate to match training ingestion convention.
        img_main = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
        img_wrist = np.ascontiguousarray(
            raw_obs["robot0_eye_in_hand_image"][::-1, ::-1]
        )
        img_main = resize_image(img_main, self.image_size).astype(np.uint8)
        img_wrist = resize_image(img_wrist, self.image_size).astype(np.uint8)

        # 8-D state: pos(3) + axis-angle(3) + gripper_qpos(2)
        pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
        quat = np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32)
        axis_angle = quat2axisangle(quat).astype(np.float32)
        gripper = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)
        state = np.concatenate([pos, axis_angle, gripper], axis=0).astype(np.float32)

        return {
            "images": [img_main, img_wrist],
            "state": state,
            "instruction": task_description,
            "dataset_name": self.normalizer_source or self.task_suite,
        }

    def format_action(self, raw_action: np.ndarray) -> Any:
        """把一个 7-D action 向量转成 ``env.step()`` 接受的参数。

        Server 端已经把 pos/rot delta 反归一化成了原始的 delta-EEF 单位，
        直接透传即可。**gripper** 维 (action[6]) 需要做一次语义重映射 ——
        而且这个映射的符号和从 dataset 表面上看到的正好相反。

        为什么两侧不一致：值域不同，符号相反
        --------------------------------------
        Dataset 的 gripper 列是 ``{0.0, 1.0}``（二值）。Env.step 的 gripper
        指令是 ``[-1.0, +1.0]``（连续）。它们其实是**同一个物理动作的两种
        不同表示**，中间被 openvla 数据管线重标签过一次。两个值域属于不同
        层级，是历史/习惯差异，不是 bug。

        1) Env 侧 —— ``[-1, +1]`` 是 robosuite 的连续执行器信号

           LIBERO 的 ``env.step`` 底层是 robosuite 的 ``GripperController``，
           它期望一个连续的 actuation 信号 ∈ ``[-1, +1]``：

               +1.0  =  最大闭合力  →  物理 CLOSE
               -1.0  =  最大张开力  →  物理 OPEN
                0.0  =  不加力（no-op）
                0.5  =  半速闭合（也是合法的）

           符号代表力的方向，数值代表力的大小。``{-1, +1}`` 只是把它饱和
           成二元动作，本质上还是一个连续控制量。这是 robosuite 的通用约定
           （所有 robosuite gripper 都长这样），不是 LIBERO 的选择。

        2) Dataset 侧 —— ``{0, 1}`` 是 openvla 重标签过的版本

           原始 ``modified_libero_rlds`` 里存的是**连续的 robosuite 命令**
           （演示者用空间鼠标采的，跨整个 [-1, +1]）。openvla 在转 RLDS 时
           一步做了两件事：

             a) 二值化：扔掉中间值，只保留"开 / 关"两个离散标签
                （反正演示者基本都打到极限，几乎没丢信息）。
             b) 翻符号 + 搬到 [0, 1]：
                    raw_cmd < 0  (OPEN,  旧 -1)  →  1.0
                    raw_cmd ≥ 0  (CLOSE, 旧 +1)  →  0.0

           翻符号是为了让标签更像一个"开关布尔值"（``1 = 开``、``0 = 关``）
           而不是"电机力符号"，这样搭 sigmoid / BCE head 更自然，也和其它
           已经归一化到 ``[0, 1]`` 的 dataset 特征统一。

           openpi 的 ``convert_libero_data_to_lerobot.py`` 原样透传，所以
           我们训练用的 LeRobot parquet 里继承了这个"翻了符号 + 二值化"
           的约定。实测结果：

               action[6] = 1.0  →  物理 OPEN   (aperture 增大)
               action[6] = 0.0  →  物理 CLOSE  (aperture 减小)

           模型用 ``binary`` normalization 模式训练，学到的输出 ≈ 0 或 ≈ 1。
           Server 端对 ``binary`` 模式的反归一化是 identity，所以到 client
           这里拿到的仍是一个接近 0 或 1 的连续 float —— **不是**一个合法
           的 ``env.step`` gripper 指令。

        一句话总结
        ----------
        * Env 要 ``[-1, +1]`` 因为它本来是**连续电机力**，符号代表方向。
        * Dataset 是 ``{0, 1}`` 因为 openvla **故意重标签**，让标签更像
          "开关布尔"而不是电机力。
        * 两者符号正好相反，所以 ``format_action`` 必须做 ``> 0.5 → -1``
          （而不是 ``+1``）。

        LIBERO-plus 的 Sylvest/LeRobot 数据不同：``action[6]`` 已经是
        robosuite env command 的 ``[-1, +1]`` 表示，不是 OpenVLA 的
        ``{0, 1}`` 标签，所以 ``normalizer_source=libero_plus`` 时不能翻符号。

        """
        action = np.asarray(raw_action, dtype=np.float64).flatten()
        if action.size < 7:
            raise ValueError(
                f"LiberoEnv.format_action expected >=7-D action, got shape {action.shape}"
            )

        if self.normalizer_source == "libero_plus":
            action[6] = float(np.clip(action[6], -1.0, 1.0))
            return action.tolist()

        # Gripper 二值化。注意：dataset 标签和 env.step 指令的符号是**相反**
        # 的（详见上面 docstring）—— dataset 1.0=open 对应 env cmd -1.0=open，
        # dataset 0.0=close 对应 env cmd +1.0=close。别"好心"改回 +1.0。
        action[6] = -1.0 if action[6] > 0.5 else 1.0

        return action.tolist()

    def get_render_frame(self, env: Any, raw_obs: Any) -> np.ndarray:
        return raw_obs["agentview_image"][::-1, ::-1]
