"""Base class for simulation-based evaluation environments.

Provides multi-process episode evaluation with video recording.
Subclasses implement environment-specific logic (make_env, get_observation,
format_action, task_count).

No torch dependency — works with PolicyClient over WebSocket.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    task_id: int
    episode_idx: int
    success: bool
    reward: float


class BaseSimEnv(ABC):
    """Abstract base for simulation evaluation environments.

    Subclasses must implement 4 methods:
        make_env, get_observation, format_action, task_count
    """

    # --- Configurable attributes (override in subclass) ---
    max_steps: int = 300
    num_steps_wait: int = 0
    dummy_action: Any = None

    # --- Abstract methods ---

    @abstractmethod
    def make_env(self, task_id: int, seed: int) -> Tuple[Any, str]:
        """Create an environment instance and return (env, task_description)."""
        ...

    @abstractmethod
    def get_observation(self, env: Any, raw_obs: Any, task_description: str) -> dict:
        """Convert raw env observation to model input dict (numpy arrays/strings)."""
        ...

    @abstractmethod
    def format_action(self, raw_action: np.ndarray) -> Any:
        """Post-process model output (e.g. denormalize) for env.step()."""
        ...

    @abstractmethod
    def task_count(self) -> int:
        """Number of tasks in the benchmark."""
        ...

    def task_descriptions(self) -> Optional[List[str]]:
        """Return human-readable task descriptions (one per task), or None."""
        return None

    def reset_env(
        self, env: Any, task_id: int, episode_idx: int
    ) -> Any:
        """Reset the env for a new episode and return the first raw observation.

        Default implementation just calls ``env.reset()``. Subclasses may
        override this to apply benchmark-specific initial states (e.g., the
        LIBERO benchmark provides per-task ``init_states``).
        """
        return env.reset()

    # --- Video helper (optional override) ---

    def get_render_frame(self, env: Any, raw_obs: Any) -> np.ndarray:
        """Return an RGB frame for video recording. Override for custom rendering."""
        return np.zeros((256, 256, 3), dtype=np.uint8)

    # --- Public API ---

    def evaluate(
        self,
        policy,
        num_episodes_per_task: int = 50,
        num_workers: int = 8,
        video_dir: Optional[str] = None,
        seed: int = 42,
        max_tasks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run full evaluation across all tasks.

        Args:
            policy: A PolicyClient with predict_action().
            num_episodes_per_task: Episodes to run per task.
            num_workers: Number of parallel worker processes.
            video_dir: If set, save episode videos here.
            seed: Base random seed.
            max_tasks: Debug-only cap on the first N tasks. ``None`` evaluates all.

        Returns:
            Dict with per-task success rates and overall average.
        """
        from sim_eval.client import PolicyClient

        n_tasks = self.task_count()

        if max_tasks is None or max_tasks <= 0:
            task_ids = list(range(n_tasks))
        else:
            task_ids = list(range(min(int(max_tasks), n_tasks)))
            logger.warning(
                "Debug task limit enabled: evaluating first %d/%d tasks.",
                len(task_ids), n_tasks,
            )

        # Log task descriptions for visibility
        descs = self.task_descriptions()
        if descs:
            if len(task_ids) == n_tasks:
                logger.info("Tasks (%d):", n_tasks)
            else:
                logger.info("Tasks (%d/%d):", len(task_ids), n_tasks)
            for task_id in task_ids:
                logger.info("  [%d] %s", task_id, descs[task_id])

        episodes = [
            (task_id, ep_idx)
            for task_id in task_ids
            for ep_idx in range(num_episodes_per_task)
        ]

        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

        use_mp = isinstance(policy, PolicyClient) and num_workers > 1

        if use_mp:
            results = self._evaluate_multiprocess(
                policy, episodes, num_workers, video_dir, seed,
            )
        else:
            results = self._evaluate_single(policy, episodes, video_dir, seed)

        return self._aggregate_results(
            results, n_tasks, num_episodes_per_task, task_ids=task_ids,
        )

    # --- Internal: multi-process ---

    def _evaluate_multiprocess(
        self,
        policy,
        episodes: List[Tuple[int, int]],
        num_workers: int,
        video_dir: Optional[str],
        seed: int,
    ) -> List[EpisodeResult]:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        # Distribute episodes in task-contiguous blocks so each worker
        # handles consecutive episodes of the same task and can reuse the env.
        worker_episodes = [[] for _ in range(num_workers)]
        for i, ep in enumerate(episodes):
            worker_episodes[i % num_workers].append(ep)
        # Sort each worker's episodes by task_id so same-task episodes are adjacent.
        for w in worker_episodes:
            w.sort(key=lambda x: x[0])

        # Extract connection info from policy client
        host = policy.uri.replace("ws://", "").split(":")[0]
        port = int(policy.uri.replace("ws://", "").split(":")[1])

        processes = []
        for worker_id in range(num_workers):
            if not worker_episodes[worker_id]:
                continue
            p = ctx.Process(
                target=_worker_fn,
                args=(
                    self,
                    host,
                    port,
                    worker_episodes[worker_id],
                    video_dir,
                    seed,
                    result_queue,
                    worker_id,  # for MUJOCO_EGL_DEVICE_ID pinning via LEAP_SIM_GPUS
                ),
                daemon=True,  # die with parent (SIGINT, SIGTERM, or exit)
            )
            p.start()
            processes.append(p)

        # Collect results with progress bar.
        results: List[EpisodeResult] = []
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(episodes), desc="Evaluating")
        except ImportError:
            pbar = None

        expected = len(episodes)
        interrupted = False
        try:
            while len(results) < expected:
                try:
                    # Short timeout so we can detect (a) Ctrl+C quickly
                    # and (b) all-workers-dead without blocking forever.
                    result = result_queue.get(timeout=2.0)
                except Exception:  # Empty + KeyboardInterrupt catch-all
                    # If every worker has exited (e.g. server died, all
                    # ran out of episodes) there's no one to produce
                    # more results — stop draining.
                    alive = [p for p in processes if p.is_alive()]
                    if not alive:
                        logger.warning(
                            "All %d workers exited before all episodes finished "
                            "(%d/%d collected). Stopping.",
                            len(processes), len(results), expected,
                        )
                        break
                    continue
                results.append(result)
                if pbar is not None:
                    pbar.update(1)
        except KeyboardInterrupt:
            interrupted = True
            logger.warning("KeyboardInterrupt received — terminating workers.")
        finally:
            if pbar is not None:
                pbar.close()

            # Terminate + reap. With daemon=True children die automatically
            # when the parent exits, but we still want tidy shutdown.
            for p in processes:
                if p.is_alive():
                    p.terminate()
            for p in processes:
                p.join(timeout=5)
                if p.is_alive():
                    # Last resort: SIGKILL
                    try:
                        p.kill()
                    except Exception:
                        pass

        if interrupted:
            raise KeyboardInterrupt

        return results

    def _evaluate_single(
        self,
        policy,
        episodes: List[Tuple[int, int]],
        video_dir: Optional[str],
        seed: int,
    ) -> List[EpisodeResult]:
        results = []
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(episodes), desc="Evaluating")
        except ImportError:
            pbar = None

        for task_id, ep_idx in episodes:
            result = _run_episode(self, policy, task_id, ep_idx, video_dir, seed)
            results.append(result)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        return results

    def _aggregate_results(
        self,
        results: List[EpisodeResult],
        n_tasks: int,
        num_episodes_per_task: int,
        task_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        eval_task_ids = task_ids if task_ids is not None else list(range(n_tasks))
        per_task: Dict[int, List[bool]] = {i: [] for i in eval_task_ids}
        for r in results:
            per_task.setdefault(r.task_id, []).append(r.success)

        # Use task descriptions (e.g. "beat_block_hammer", or LIBERO's
        # natural-language task name) as keys so the JSON / log output
        # mirrors starvla README tables. Falls back to ``task_NNN`` if the
        # subclass doesn't expose ``task_descriptions()`` or returns the
        # wrong number.
        descs = self.task_descriptions()
        use_named_keys = descs is not None and len(descs) == n_tasks

        task_success_rates = {}
        for task_id in eval_task_ids:
            successes = per_task[task_id]
            rate = sum(successes) / len(successes) if successes else 0.0
            key = descs[task_id] if use_named_keys else f"task_{task_id:03d}"
            task_success_rates[key] = rate

        overall = (
            sum(task_success_rates.values()) / len(task_success_rates)
            if task_success_rates else 0.0
        )

        logger.info("Overall success rate: %.1f%%", overall * 100)
        for name, rate in task_success_rates.items():
            logger.info("  %s: %.1f%%", name, rate * 100)

        output = {
            "overall_success_rate": overall,
            "per_task_success_rate": task_success_rates,
            "num_episodes_per_task": num_episodes_per_task,
            "total_episodes": len(results),
            "num_tasks_evaluated": len(eval_task_ids),
            "num_tasks_total": n_tasks,
        }
        if len(eval_task_ids) != n_tasks:
            output["task_ids"] = eval_task_ids
        return output


# --- Module-level functions (picklable for multiprocessing) ---


class _ServerUnavailable(Exception):
    """Raised when the policy server has gone away mid-episode.

    Distinct from generic ``Exception`` so the worker can tell the
    difference between a *one-off simulator error* (carry on to the next
    episode) and a *terminal server failure* (give up — no point trying
    the remaining episodes).
    """


def _install_signal_handlers() -> None:
    """Make Ctrl+C kill the worker immediately (SIGTERM handler exits cleanly).

    ``spawn`` multiprocessing does NOT inherit the parent's SIGINT handler,
    so by default workers keep running after the parent dies. We install
    a no-op KeyboardInterrupt handler so that ``terminate()`` in the
    parent works cleanly, and a SIGTERM handler that exits the worker.
    """
    import signal as _signal
    try:
        _signal.signal(_signal.SIGINT, _signal.SIG_IGN)   # let parent drive shutdown
        _signal.signal(_signal.SIGTERM, lambda *_: os._exit(0))
    except (ValueError, OSError):
        pass  # signals can't always be set on every platform / thread


def _pin_mujoco_gpu(worker_id: int) -> None:
    """Pin this worker's MuJoCo offscreen renderer to a specific GPU.

    Reads ``LEAP_SIM_GPUS`` (comma-separated ids, e.g. ``"5,6,7"``) from the
    environment and assigns worker ``i`` to GPU ``sim_gpus[i % len(sim_gpus)]``
    via ``MUJOCO_EGL_DEVICE_ID`` (the env var MuJoCo's EGL backend reads).

    This **must** happen before any ``mujoco`` / ``robosuite`` import in the
    worker, otherwise the GL context is pinned to whatever GPU index MuJoCo
    autodiscovers on first import (usually 0, which conflicts with the
    policy server if it also lives on GPU 0).

    If ``LEAP_SIM_GPUS`` is unset, we leave MuJoCo's default behaviour
    (GPU 0). The shell script (``scripts/eval_libero.sh``) surfaces this
    as the ``--sim-gpus`` knob; users SHOULD set it to avoid colliding
    with ``leap serve --gpu``.
    """
    raw = os.environ.get("LEAP_SIM_GPUS", "").strip()
    if not raw:
        return
    try:
        gpus = [int(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        logger.warning("LEAP_SIM_GPUS=%r is not a comma-separated int list; ignoring.", raw)
        return
    if not gpus:
        return
    my_gpu = gpus[worker_id % len(gpus)]
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(my_gpu)
    # Some MuJoCo versions also read CUDA_VISIBLE_DEVICES to pick the GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(my_gpu)
    logger.info("worker %d → MUJOCO_EGL_DEVICE_ID=%d", worker_id, my_gpu)


def _silence_sim_warnings() -> None:
    """Silence the noisy third-party warnings emitted on every worker startup.

    Specifically suppresses:

    * ``[robosuite WARNING] No private macro file found!`` etc.
    * ``Gym has been unmaintained since 2022`` deprecation banner.
    * ``[Warning]: datasets path ... does not exist!`` from LIBERO's
      ``__init__.py``. (Already wrapped at call site, but belt-and-suspenders
      here so even a top-level re-import stays quiet.)
    * Python ``DeprecationWarning`` / ``UserWarning`` spam.
    * ``Warp DeprecationWarning: warp.torch ...`` re-emitted on every
      ``import curobo`` inside RoboTwin's per-task setup_demo. Unlike most
      libs, NVIDIA Warp prints DeprecationWarnings via Python's standard
      ``warnings.warn``, but it does so on *every import* of certain
      submodules — so we explicitly add a regex filter targeting them.
    * ``Exception ignored in: <function MjRenderContext.__del__ ...>`` — a
      cosmetic AttributeError from robosuite's renderer finalizer; we
      monkey-patch the offending ``__del__`` to swallow it.

    Disabled by ``LIBERO_SUPPRESS_INIT_WARN=0`` for debugging. Does NOT
    touch ``ERROR``-level logs — real problems still surface.
    """
    import warnings
    import logging as _logging

    if os.environ.get("LIBERO_SUPPRESS_INIT_WARN", "1") == "0":
        return

    warnings.filterwarnings("ignore")
    # Specifically silence NVIDIA Warp's per-import deprecation spam
    # (printed by curobo on every RoboTwin task class import). Use both
    # category-name regex and module-name regex so we catch it whether
    # warp uses category="Warp DeprecationWarning" (custom subclass) or
    # the standard DeprecationWarning category.
    warnings.filterwarnings("ignore", message=r".*Warp Deprecation.*")
    warnings.filterwarnings("ignore", message=r".*warp\.torch.*")
    warnings.filterwarnings("ignore", message=r".*device_from_torch.*")
    warnings.filterwarnings("ignore", module=r".*warp.*")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    # robosuite + gym + mujoco use Python logging; silence everything below ERROR.
    for name in ("robosuite", "gym", "gymnasium", "mujoco", "mujoco_py", "libero",
                 "warp", "curobo", "sapien", "mplib", "toppra"):
        lg = _logging.getLogger(name)
        lg.setLevel(_logging.ERROR)
        lg.propagate = False

    # Best-effort patch for robosuite's finalizer noise. Lazy — only if
    # the module is already importable; failure here is fine, the
    # "Exception ignored" message is cosmetic anyway.
    # We import robosuite under a stdout+stderr redirect so the import
    # banners don't print when this helper runs.
    try:  # pragma: no cover — environment-specific
        import contextlib as _contextlib, io as _io
        with _contextlib.redirect_stdout(_io.StringIO()), _contextlib.redirect_stderr(_io.StringIO()):
            from robosuite.utils import binding_utils as _bu
        if hasattr(_bu, "MjRenderContext"):
            _orig_del = _bu.MjRenderContext.__del__

            def _quiet_del(self, _orig=_orig_del):
                try:
                    _orig(self)
                except AttributeError:
                    pass

            _bu.MjRenderContext.__del__ = _quiet_del
    except Exception:
        pass


def _worker_fn(
    sim_env: BaseSimEnv,
    host: str,
    port: int,
    episodes: List[Tuple[int, int]],
    video_dir: Optional[str],
    seed: int,
    result_queue: mp.Queue,
    worker_id: int = 0,
) -> None:
    """Worker process: create own :class:`PolicyClient`, run assigned episodes.

    Reuses the env across consecutive episodes of the same task to avoid the
    overhead of repeated env creation/destruction (LIBERO env init is expensive).

    If the server goes away mid-run we DON'T retry every episode and spam
    the same error — we emit a best-effort "failed" result for each
    remaining (task, episode) pair and exit. The parent process reads
    these placeholder results and aggregates normally.
    """
    # IMPORTANT: pin GPU + silence warnings BEFORE any libero / robosuite
    # import. Order matters — MuJoCo's EGL backend reads
    # MUJOCO_EGL_DEVICE_ID on first use, so pinning after the import would
    # be a no-op.
    _pin_mujoco_gpu(worker_id)
    _silence_sim_warnings()
    _install_signal_handlers()

    from sim_eval.client import PolicyClient

    client = PolicyClient(host, port)
    aborted = False
    current_env = None
    current_task_id = None
    current_task_desc = None
    try:
        for task_id, ep_idx in episodes:
            if aborted:
                # Push placeholder fail results so the parent's queue
                # drain finishes in bounded time.
                result_queue.put(
                    EpisodeResult(task_id=task_id, episode_idx=ep_idx, success=False, reward=0.0)
                )
                continue
            try:
                # Reuse env if same task, otherwise create a new one
                if task_id != current_task_id:
                    if current_env is not None and hasattr(current_env, "close"):
                        try:
                            current_env.close()
                        except Exception:
                            pass
                    ep_seed = seed + task_id * 10000 + ep_idx
                    current_env, current_task_desc = sim_env.make_env(task_id, ep_seed)
                    current_task_id = task_id

                result = _run_episode_reuse_env(
                    sim_env, client, current_env, current_task_desc,
                    task_id, ep_idx, video_dir, seed,
                )
            except _ServerUnavailable as exc:
                logger.error(
                    "Server gone at task=%d ep=%d (%s) — "
                    "aborting remaining episodes in this worker.",
                    task_id, ep_idx, exc,
                )
                aborted = True
                result = EpisodeResult(
                    task_id=task_id, episode_idx=ep_idx, success=False, reward=0.0
                )
            except Exception:
                # Env might be in a bad state — force recreate on next iteration
                logger.exception("Episode failed: task=%d ep=%d", task_id, ep_idx)
                current_task_id = None
                if current_env is not None and hasattr(current_env, "close"):
                    try:
                        current_env.close()
                    except Exception:
                        pass
                current_env = None
                result = EpisodeResult(
                    task_id=task_id, episode_idx=ep_idx, success=False, reward=0.0
                )
            result_queue.put(result)
    finally:
        if current_env is not None and hasattr(current_env, "close"):
            try:
                current_env.close()
            except Exception:
                pass
        client.close()


def _run_episode_reuse_env(
    sim_env: BaseSimEnv,
    policy,
    env: Any,
    task_desc: str,
    task_id: int,
    ep_idx: int,
    video_dir: Optional[str],
    seed: int,
) -> EpisodeResult:
    """Run a single episode reusing an existing env (avoids make_env overhead).

    Same logic as _run_episode but skips env creation/destruction.
    """
    chunk: Optional[np.ndarray] = None
    chunk_idx = 0

    def _next_action(obs_dict: dict) -> np.ndarray:
        nonlocal chunk, chunk_idx
        if chunk is None or chunk_idx >= chunk.shape[0]:
            raw = policy.predict_action(**obs_dict)
            raw = np.asarray(raw)
            if raw.ndim == 3:
                raw = raw[0]
            if raw.ndim == 2:
                env_cap = getattr(sim_env, "action_chunk_size", None)
                if env_cap is None or env_cap <= 0:
                    max_chunk = raw.shape[0]
                else:
                    max_chunk = min(int(env_cap), raw.shape[0])
                chunk = raw[: max_chunk]
                chunk_idx = 0
            else:
                chunk = raw[None, :]
                chunk_idx = 0
        act = chunk[chunk_idx]
        chunk_idx += 1
        return act

    info: dict = {}
    try:
        raw_obs = sim_env.reset_env(env, task_id, ep_idx)
        frames = []
        total_reward = 0.0
        success = False
        done = False

        for step in range(sim_env.max_steps + sim_env.num_steps_wait):
            if video_dir is not None:
                frames.append(sim_env.get_render_frame(env, raw_obs))

            if step < sim_env.num_steps_wait:
                if sim_env.dummy_action is not None:
                    raw_obs, reward, done, info = env.step(sim_env.dummy_action)
                continue

            obs = sim_env.get_observation(env, raw_obs, task_desc)
            try:
                raw_action = _next_action(obs)
            except Exception as exc:
                if _is_server_gone_error(exc):
                    raise _ServerUnavailable(str(exc)) from exc
                raise
            action = sim_env.format_action(raw_action)

            raw_obs, reward, done, info = env.step(action)
            total_reward += float(reward)

            if done:
                success = bool(info.get("success", reward > 0))
                break

        if not done:
            success = bool(info.get("success", False)) if info else False

        if video_dir is not None and frames:
            _save_video(frames, video_dir, task_id, ep_idx, success)

    except _ServerUnavailable:
        raise
    except Exception:
        logger.exception("Episode failed: task=%d ep=%d", task_id, ep_idx)
        raise  # let worker_fn handle env recreation

    return EpisodeResult(
        task_id=task_id,
        episode_idx=ep_idx,
        success=success,
        reward=total_reward,
    )


def _run_episode(
    sim_env: BaseSimEnv,
    policy,
    task_id: int,
    ep_idx: int,
    video_dir: Optional[str],
    seed: int,
) -> EpisodeResult:
    """Run a single evaluation episode.

    Supports **action chunking**: if ``policy.predict_action(...)`` returns a
    2-D array of shape ``(T, D_a)`` we dispatch the ``T`` actions one at a time
    and only re-query the server every ``chunk_size`` env steps. This mirrors
    starVLA's ``ModelClient.step`` behaviour and is essential for flow-matching
    heads where each server call runs a multi-step denoising loop.

    Chunk sizing: we follow ``sim_env.action_chunk_size`` (falls back to the
    full chunk returned by the server) and cap at the number of future actions
    actually produced by the model.
    """
    ep_seed = seed + task_id * 10000 + ep_idx

    # Env construction is fragile (robosuite renderer can fail for a dozen
    # reasons: missing MuJoCo GL, incompatible robosuite version, bad bddl
    # path, …). When it does, the follow-up GC of MjRenderContext prints a
    # misleading ``AttributeError: ... 'con'`` that hides the real cause.
    # Catch + log once here so the user sees the actual stack trace, and
    # return a placeholder fail result so the worker can move on.
    try:
        env, task_desc = sim_env.make_env(task_id, ep_seed)
    except Exception as exc:
        if _is_server_gone_error(exc):
            raise _ServerUnavailable(str(exc)) from exc
        logger.exception(
            "make_env failed: task=%d ep=%d — skipping. "
            "If every episode hits this, re-run with --workers 1 for a "
            "readable traceback.",
            task_id, ep_idx,
        )
        return EpisodeResult(task_id=task_id, episode_idx=ep_idx, success=False, reward=0.0)

    # Chunk cache — populated lazily on first policy call.
    chunk: Optional[np.ndarray] = None
    chunk_idx = 0

    def _next_action(obs_dict: dict) -> np.ndarray:
        nonlocal chunk, chunk_idx
        if chunk is None or chunk_idx >= chunk.shape[0]:
            raw = policy.predict_action(**obs_dict)
            raw = np.asarray(raw)
            if raw.ndim == 3:
                # (B=1, T, D_a) → (T, D_a)
                raw = raw[0]
            if raw.ndim == 2:
                # Default chunking: use the full chunk the server returned
                # (aligns with training ``action_horizon``). Only cap if the
                # env explicitly requested a shorter window.
                env_cap = getattr(sim_env, "action_chunk_size", None)
                if env_cap is None or env_cap <= 0:
                    max_chunk = raw.shape[0]
                else:
                    max_chunk = min(int(env_cap), raw.shape[0])
                chunk = raw[: max_chunk]
                chunk_idx = 0
            else:
                # single action — synthesize a 1-step chunk
                chunk = raw[None, :]
                chunk_idx = 0
        act = chunk[chunk_idx]
        chunk_idx += 1
        return act

    info: dict = {}
    try:
        raw_obs = sim_env.reset_env(env, task_id, ep_idx)
        frames = []
        total_reward = 0.0
        success = False
        done = False

        for step in range(sim_env.max_steps + sim_env.num_steps_wait):
            if video_dir is not None:
                frames.append(sim_env.get_render_frame(env, raw_obs))

            # Wait steps: apply dummy action
            if step < sim_env.num_steps_wait:
                if sim_env.dummy_action is not None:
                    raw_obs, reward, done, info = env.step(sim_env.dummy_action)
                continue

            # Get model observation + predict (with chunking).
            # ConnectionClosed from the policy server is a *terminal*
            # signal — bubble it up as _ServerUnavailable so the worker
            # stops trying further episodes.
            obs = sim_env.get_observation(env, raw_obs, task_desc)
            try:
                raw_action = _next_action(obs)
            except Exception as exc:
                if _is_server_gone_error(exc):
                    raise _ServerUnavailable(str(exc)) from exc
                raise
            action = sim_env.format_action(raw_action)

            raw_obs, reward, done, info = env.step(action)
            total_reward += float(reward)

            if done:
                success = bool(info.get("success", reward > 0))
                break

        if not done:
            success = bool(info.get("success", False)) if info else False

        # Save video
        if video_dir is not None and frames:
            _save_video(frames, video_dir, task_id, ep_idx, success)

    except _ServerUnavailable:
        # Re-raise — the worker loop will stop processing further episodes.
        raise
    except Exception:
        logger.exception("Episode failed: task=%d ep=%d", task_id, ep_idx)
        success = False
        total_reward = 0.0
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    return EpisodeResult(
        task_id=task_id,
        episode_idx=ep_idx,
        success=success,
        reward=total_reward,
    )


def _is_server_gone_error(exc: BaseException) -> bool:
    """Return True if *exc* indicates the policy server closed the socket.

    We match by class name (via MRO) so we don't depend on a specific
    ``websockets`` version.
    """
    for cls in type(exc).__mro__:
        name = cls.__name__
        if name in {"ConnectionClosed", "ConnectionClosedOK", "ConnectionClosedError"}:
            return True
    # socket-level failures also count
    if isinstance(exc, (ConnectionError, BrokenPipeError, TimeoutError, OSError)):
        return True
    return False


def _save_video(
    frames: List[np.ndarray],
    video_dir: str,
    task_id: int,
    ep_idx: int,
    success: bool,
) -> None:
    """Save episode frames as MP4."""
    try:
        import imageio
        tag = "ok" if success else "fail"
        path = os.path.join(video_dir, f"{task_id:03d}_{ep_idx:03d}_{tag}.mp4")
        imageio.mimwrite(path, frames, fps=30)
    except ImportError:
        logger.warning("imageio not installed — skipping video save")
    except Exception:
        logger.exception("Failed to save video for task=%d ep=%d", task_id, ep_idx)
