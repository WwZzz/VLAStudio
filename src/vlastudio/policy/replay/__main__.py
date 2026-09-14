"""python -m policy.replay --list-episodes local.t0325"""

import argparse

from vlastudio.policy.replay.dataset_episode import list_episode_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay policy helpers")
    ap.add_argument("--list-episodes", metavar="TASK", required=True, help="e.g. local.t0325")
    ns, rest = ap.parse_known_args()
    ids = list_episode_ids(ns.list_episodes, unknown_args=rest)
    print(f"task={ns.list_episodes!r} num_episodes={len(ids)}")
    print("episode_id values:", ids)


if __name__ == "__main__":
    main()
