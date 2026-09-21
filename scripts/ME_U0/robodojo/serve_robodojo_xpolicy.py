"""Serve a trained ME_U0 checkpoint with RoboDojo's XPolicyLab WS protocol."""

from __future__ import annotations

import argparse
import asyncio
import os
import random

import numpy as np
import torch


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer_per_view_size(cfg, default: int = 96) -> int | tuple[int, int]:
    for dataset in cfg["data"]["train"].get("datasets", []) or []:
        resize = dataset.get("decode_resize")
        if resize:
            if len(resize) != 2:
                raise ValueError(
                    "ME_U0 serving expects decode_resize=[height,width], "
                    f"got {resize}"
                )
            height, width = int(resize[0]), int(resize[1])
            if height <= 0 or width <= 0:
                raise ValueError(f"decode_resize must be positive, got {resize}")
            return height if height == width else (height, width)
    return default


def _infer_mosaic_layout(cfg) -> str:
    for dataset in cfg["data"]["train"].get("datasets", []) or []:
        layout = dataset.get("concat_multi_camera", "horizontal")
        if layout in ("pyramid_3cam", "robotwin"):
            return "pyramid"
        if layout == "horizontal":
            return "horizontal"
        raise ValueError(f"Unsupported RoboDojo serving camera mosaic: {layout!r}")
    return "horizontal"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--stats",
        default=None,
        help=(
            "override cfg.data.train.datasets[*].pretrained_norm_stats for "
            "the single RoboDojo codec"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="RoboDojo dataset codec; inferred when the config contains exactly one",
    )
    parser.add_argument("--domain-id", type=int, default=13)
    parser.add_argument("--action-chunk-size", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--per-view-size", type=int, default=None)
    parser.add_argument(
        "--mosaic-layout",
        choices=("auto", "horizontal", "pyramid"),
        default="auto",
        help=(
            "camera mosaic layout; auto reads the training config, "
            "pyramid uses one full-resolution head view above two "
            "half-resolution wrist views"
        ),
    )
    parser.add_argument("--max-text-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-task",
        default=None,
        help="RoboDojo task associated with this per-task policy server",
    )
    parser.add_argument(
        "--clip-normalized-actions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "override config clipping before inverse normalization; the clip "
            "magnitude comes from robodojo_normalized_action_clip_value"
        ),
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _seed_everything(args.seed)

    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    from leap.core.config import instantiate, load_config
    from leap.data.world_unified.metadata import metadata_by_dataset_from_config
    from leap.models.builder import load_pretrained_weights
    from leap.serving.ME_U0_policy import MachEmbodiedUnifiedPolicy
    from leap.serving.me_u0_robodojo_codec import build_robodojo_codecs_from_config
    from scripts.ME_U0.robodojo.robodojo_xpolicy_policy import RoboDojoXPolicyModel

    cfg = load_config(args.config, overrides=args.overrides)
    mosaic_layout = (
        _infer_mosaic_layout(cfg)
        if args.mosaic_layout == "auto"
        else args.mosaic_layout
    )
    if mosaic_layout == "pyramid" and args.per_view_size is not None:
        parser.error(
            "--per-view-size is a legacy square override; omit it for pyramid "
            "so [height,width] is read from the experiment config"
        )
    metadata = metadata_by_dataset_from_config(cfg)
    clip_normalized_actions = args.clip_normalized_actions
    if clip_normalized_actions is None:
        clip_normalized_actions = bool(
            cfg.get("robodojo_clip_normalized_actions", True)
        )
    codecs = build_robodojo_codecs_from_config(
        cfg,
        clip_normalized_actions=clip_normalized_actions,
        stats_path_override=args.stats,
    )
    dataset_name = args.dataset_name
    if dataset_name is None:
        if len(codecs) != 1:
            raise ValueError(
                "--dataset-name is required when a config contains multiple "
                f"RoboDojo datasets: {list(codecs)}"
            )
        dataset_name = next(iter(codecs))
    if dataset_name not in codecs:
        raise KeyError(f"dataset_name={dataset_name!r} not in codecs={list(codecs)}")
    codec = codecs[dataset_name]
    print(f"[RoboDojo] codec_contract={codec.contract}", flush=True)

    model = instantiate(cfg.get("model")).to(torch.bfloat16)
    print(f"[RoboDojo] loading strict checkpoint: {args.checkpoint}", flush=True)
    load_pretrained_weights(model, args.checkpoint, strict=True)
    model = model.to(args.device).eval()

    per_view_size = args.per_view_size or _infer_per_view_size(cfg)
    policy = MachEmbodiedUnifiedPolicy(
        model,
        per_view_size=per_view_size,
        mosaic_layout=mosaic_layout,
        quantize_images=True,
        raw_action_dim=codec.canonical_action_dim,
        num_inference_steps=args.num_inference_steps,
        domain_id=args.domain_id,
        max_text_len=args.max_text_len,
        device=args.device,
        action_codecs=codecs,
        domain_ids={dataset_name: args.domain_id},
        metadata=metadata,
    )
    xpolicy_model = RoboDojoXPolicyModel(
        policy,
        dataset_name=dataset_name,
        action_chunk_size=args.action_chunk_size,
        action_type=codec.action_type,
    )
    server = PolicyServer(
        xpolicy_model,
        PolicyServerConfig(
            host=args.host,
            port=args.port,
            # Batch evaluation may spend longer than the websocket default
            # timeout executing an action chunk while the client event loop is
            # not running.  Disable protocol keepalive on both endpoints; the
            # lane watchdog remains responsible for detecting stalled tasks.
            ws_ping_interval_s=None,
            ws_ping_timeout_s=None,
        ),
    )
    print(
        f"[RoboDojo] WS server=ws://{args.host}:{args.port} domain={args.domain_id} "
        f"task={args.eval_task} seed={args.seed} "
        f"views={per_view_size} mosaic={mosaic_layout} "
        f"chunk={args.action_chunk_size} "
        f"action_type={codec.action_type} "
        f"normalized_action_clip={codec.contract.normalized_action_clip_value}",
        flush=True,
    )
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
