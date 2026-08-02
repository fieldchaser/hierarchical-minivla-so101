#!/usr/bin/env python3
"""Validate one saved RGB-language-action episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hierarchical_minivla.vision_data import validate_vision_episode_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()

    episode_path = args.episode.expanduser().resolve()
    with np.load(episode_path, allow_pickle=False) as data:
        summary = validate_vision_episode_arrays(data)
        summary["success"] = bool(data["success"])
    summary["episode"] = str(episode_path)
    summary["size_mb"] = round(episode_path.stat().st_size / 1_000_000, 3)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
