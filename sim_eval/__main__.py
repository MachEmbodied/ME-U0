"""LIBERO and LIBERO-plus simulation client."""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings

def _silence_cosmetic_warnings() -> None:
    """Match what ``base_env._silence_sim_warnings`` does, but BEFORE any
    robosuite / gym / libero import in the parent process."""
    if os.environ.get("LIBERO_SUPPRESS_INIT_WARN", "1") == "0":
        return
    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    for name in ("robosuite", "gym", "gymnasium", "mujoco", "mujoco_py", "libero"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.propagate = False


def main() -> None:
    _silence_cosmetic_warnings()
    parser = argparse.ArgumentParser(description="Evaluate a policy on LIBERO.")
    parser.add_argument("--env", required=True, choices=["libero"])
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--server", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--action-chunk-size", type=int, default=None)
    parser.add_argument("--normalizer-source", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    host, port = args.server.rsplit(":", 1)
    port = int(port)
    from sim_eval.libero_env import LiberoEnv
    sim_env = LiberoEnv(
        task_suite=args.task_suite, image_size=args.image_size,
        action_chunk_size=args.action_chunk_size,
        normalizer_source=args.normalizer_source,
    )

    # Build client
    from sim_eval.client import PolicyClient
    client = PolicyClient(host, port)

    try:
        results = sim_env.evaluate(
            client,
            num_episodes_per_task=args.episodes,
            num_workers=args.workers,
            video_dir=args.video_dir,
            seed=args.seed,
            max_tasks=args.max_tasks if args.max_tasks and args.max_tasks > 0 else None,
        )

        print(f"\nOverall success rate: {results['overall_success_rate']:.1%}")
        for task, rate in results["per_task_success_rate"].items():
            print(f"  {task}: {rate:.1%}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
