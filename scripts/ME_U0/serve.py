"""Standalone WebSocket policy server for MachEmbodiedUnifiedModel.

Uses the shared policy server and simulation clients.

  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/ME_U0/serve.py \
      --config leap/configs/experiments/libero_posttraining.yaml \
      --checkpoint <RUN>/checkpoints/step_N/pytorch_model/mp_rank_00_model_states.pt \
      --port 8765 --num-inference-steps 24 --seed 42
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _seed_everything(seed: int) -> None:
    """Seed every RNG used by model construction and stochastic sampling."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Match RoboDojo serving: avoid the RTX5880 compiled FlexAttention shared-memory limit.
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--max-wait-ms", type=float, default=10.0)
    ap.add_argument("--raw-action-dim", type=int, default=7)
    ap.add_argument("--per-view-size", type=int, default=None,
                    help="per-camera mosaic resolution; if unset, read from the data config's "
                         "per_view_size (train↔infer aligned), falling back to 256")
    ap.add_argument("--max-text-len", type=int, default=None,
                    help="instruction token cap; if unset, read from model.max_text_len "
                         "(train↔infer aligned), falling back to 512")
    ap.add_argument("--domain-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed Python, NumPy, and PyTorch sampling RNGs")
    ap.add_argument("--video-out-dir", default=None,
                    help="if set, save one model-generated future video per task here")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    import torch
    from leap.core.config import instantiate, load_config
    from leap.data.world_unified.metadata import metadata_by_dataset_from_config
    from leap.models.builder import load_pretrained_weights
    from leap.serving.ME_U0_policy import MachEmbodiedUnifiedPolicy
    from leap.serving.server import PolicyServer

    if args.seed is not None:
        _seed_everything(args.seed)
        print(f"[serve] seed={args.seed}", flush=True)

    cfg = load_config(args.config, overrides=args.overrides)
    if args.checkpoint:
        cfg.model.pretrained_pth = None
    model = instantiate(cfg.get("model"))
    model = model.to(torch.bfloat16)
    if args.checkpoint:
        print(f"[serve] loading checkpoint: {args.checkpoint}", flush=True)
        checkpoint = Path(args.checkpoint)
        if checkpoint.is_dir() and (checkpoint / "model.safetensors").is_file():
            checkpoint = checkpoint / "model.safetensors"
        load_pretrained_weights(model, str(checkpoint), strict=True)
    else:
        print("[serve] no --checkpoint; using freshly-initialized (random-ish) weights", flush=True)
    model = model.to(args.device).eval()

    # Build per-source normalizers + domain_ids from the data config, keyed by
    # source name (== the dataset_name the sim client sends). Train↔infer aligned
    # (same NormalizeRobotData class as the dataset).
    normalizers, domain_ids = _build_normalizers_from_config(cfg)
    if normalizers:
        print(f"[serve] normalizers for sources: {list(normalizers.keys())}", flush=True)
    metadata = metadata_by_dataset_from_config(cfg)
    if metadata:
        print(f"[serve] metadata for sources: {list(metadata.keys())}", flush=True)

    # per_view_size MUST match training. Prefer the value baked into the data
    # config (train↔infer aligned since eval uses the same --config), else CLI, else 256.
    per_view = args.per_view_size if args.per_view_size is not None else _infer_per_view_size(cfg)
    mosaic_layout = _infer_mosaic_layout(cfg)
    print(f"[serve] mosaic_layout={mosaic_layout}", flush=True)
    print(f"[serve] per_view_size={per_view}", flush=True)

    # max_text_len MUST match training too (same reasoning as per_view_size).
    max_text_len = args.max_text_len if args.max_text_len is not None else _infer_max_text_len(cfg)
    print(f"[serve] max_text_len={max_text_len}", flush=True)

    datasets = cfg["data"]["train"].get("datasets") or ()
    quantize_images = any(
        str(dataset.get("concat_multi_camera", "horizontal")).lower()
        in {"pyramid_3cam", "robotwin"}
        and any(
            token in str(dataset.get("dataset_name", "")).lower()
            for token in ("robodojo", "robotwin")
        )
        for dataset in datasets
    )
    policy = MachEmbodiedUnifiedPolicy(
        model,
        per_view_size=per_view,
        mosaic_layout=mosaic_layout,
        max_text_len=max_text_len,
        raw_action_dim=args.raw_action_dim,
        num_inference_steps=args.num_inference_steps,
        domain_id=args.domain_id,
        device=args.device,
        normalizers=normalizers,
        domain_ids=domain_ids,
        metadata=metadata,
        quantize_images=quantize_images,
        video_out_dir=args.video_out_dir,
    )
    server = PolicyServer(
        policy,
        host=args.host,
        port=args.port,
        device=args.device,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
    )
    print(f"[serve] MachEmbodiedUnifiedModel policy server on {args.host}:{args.port}", flush=True)
    server.serve_forever()


def _infer_per_view_size(cfg, default: int = 256) -> int | tuple[int, int]:
    """Read per_view_size from the first data.train source (train↔infer aligned).

    The data config bakes per_view_size into each source (make_*_sharded **kw);
    serving reuses the same --config, so reading it here keeps the mosaic
    resolution identical to training. Falls back to ``default`` if absent.
    """
    datasets = cfg["data"]["train"].get("datasets")
    if datasets:
        resize = datasets[0].get("decode_resize")
        if resize and len(resize) == 2:
            height, width = (int(resize[0]), int(resize[1]))
            return height if height == width else (height, width)
    return default


def _infer_mosaic_layout(cfg, default: str = "horizontal") -> str:
    """Read the camera mosaic layout from the first map-style data source."""
    datasets = cfg["data"]["train"].get("datasets") or ()
    if datasets:
        layout = str(datasets[0].get("concat_multi_camera", default)).lower()
        return "pyramid" if layout in {"pyramid_3cam", "robotwin", "pyramid"} else "horizontal"
    return default


def _infer_max_text_len(cfg, default: int = 512) -> int:
    """Read max_text_len from model config (train↔infer aligned).

    Serving reuses the same --config as training, so model.max_text_len keeps
    train/infer tokenization identical. Falls back to default if absent.
    """
    model = cfg.get("model", {})
    model_max_text_len = model.get("max_text_len", None)
    if model_max_text_len is not None:
        return int(model_max_text_len)
    return default


class _NativeMapStyleNormalizer:
    """Expose the training field normalizers through the policy's flat interface."""

    def __init__(self, source):
        from types import SimpleNamespace

        from leap.data.me_u0.utils.normalizer import LinearNormalizer, load_lerobot_v2_minmax

        self.shape_meta = source["shape_meta"]
        processor = source["processor"]
        self.normalizer = LinearNormalizer(
            self.shape_meta,
            processor["use_stepwise_action_norm"],
            processor["norm_default_mode"],
            processor.get("norm_exception_mode"),
            load_lerobot_v2_minmax(source["dataset_dirs"], self.shape_meta),
        )
        self.config = SimpleNamespace(
            action_dim=sum(int(field["shape"]) for field in self.shape_meta["action"]),
            state_dim=sum(int(field["shape"]) for field in self.shape_meta["state"]),
        )

    def _transform(self, batch, inverse):
        import torch

        result = dict(batch)
        for name, kind in (("states", "state"), ("actions", "action")):
            if name not in batch:
                continue
            fields = self.shape_meta[kind]
            parts = batch[name].split([int(field["shape"]) for field in fields], dim=-1)
            transformed = []
            for field, part in zip(fields, parts):
                norm = self.normalizer.normalizers[kind][field["key"]]
                transformed.append(norm.backward(part) if inverse else norm.forward(part))
            result[name] = torch.cat(transformed, dim=-1)
        return result

    def apply(self, batch):
        return self._transform(batch, inverse=False)

    def unapply(self, batch):
        return self._transform(batch, inverse=True)


def _build_normalizers_from_config(cfg):
    """Reuse each LIBERO source's training min/max normalizer."""
    normalizers, domain_ids = {}, {}
    for source in cfg["data"]["train"]["datasets"]:
        if source["norm_stats_source"] != "lerobot_v2_minmax":
            raise ValueError("LIBERO map-style serving requires lerobot_v2_minmax stats")
        name = source["dataset_name"]
        normalizers[name] = _NativeMapStyleNormalizer(source)
        domain_ids[name] = int(source["logical_domain_id"])
    return normalizers, domain_ids


if __name__ == "__main__":
    main()
