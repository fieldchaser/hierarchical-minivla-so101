# Hierarchical MiniVLA for SO-101

A simulation-first research project for language-conditioned, multi-task robotic manipulation. The long-term question is whether explicit skill phases improve data efficiency and generalization over a flat vision-language-action policy.

## Current milestone: goal-conditioned behavior cloning

Milestone 0 intentionally tests the smallest useful contract:

```text
robot state + three cube positions + one-hot goal + bin position -> action
```

The repository currently contains:

- a deterministic synthetic demonstration generator;
- a transparent ridge-regression behavior-cloning policy;
- tests for goal selection, training, and checkpoint round trips;
- a smoke test that instantiates and steps ETH's SO-101 multicube MuJoCo scene.

This is not yet a VLA and it does not claim task success in MuJoCo. Images, natural-language instructions, action chunking, and hierarchical skill prediction are later milestones.

## Verified result

On 2 August 2026, the baseline passed all five unit tests and reached a held-out
synthetic action MSE of `3.42e-16` with the default seed. The upstream multicube
scene also loaded successfully under MuJoCo 3.11, exposed 6 arm joints and 9
cube-position values, and advanced one physics control step. The MSE is a data-
path smoke-test metric, not a simulated pick-and-place success rate.

## Quick start

Create an environment and install this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/smoke_baseline.py
```

To check the upstream multicube simulator, first obtain the ETH course repository, install the optional simulation dependency, and pass the local HW3 path explicitly:

```bash
python -m pip install -e '.[sim]'
python scripts/smoke_eth_env.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning
```

Add `--render` when a working OpenGL display is available.
Without that flag, the script intentionally skips renderer creation and checks
only XML loading, state initialization, and physics stepping. This makes the
check usable in headless terminals such as CI.

## Why the baseline has a goal-aware feature

The task goal is a one-hot vector for red, green, or blue. Selecting the corresponding cube position is a multiplicative interaction between a goal bit and a position. A plain linear model over the raw state cannot represent that interaction. `featurize` exposes the selected target explicitly, allowing the small baseline to verify the data and conditioning path before we introduce a neural network.

The synthetic scripted expert moves toward the selected cube while the gripper is open and toward the bin while it is holding an object. It is only a supervised-learning smoke test, not a replacement for MuJoCo demonstrations.

## Roadmap

- [x] Goal-conditioned data contract and runnable BC smoke baseline
- [x] Upstream multicube environment smoke-test entry point
- [ ] Record or generate MuJoCo multicube demonstrations
- [ ] Train a neural state + one-hot goal baseline on real simulation data
- [ ] Add RGB observations and natural-language goals (Flat MiniVLA)
- [ ] Add reach/grasp/transport/release supervision (Hierarchical MiniVLA)
- [ ] Evaluate unseen layouts and instruction paraphrases

## Upstream attribution and licensing boundary

This project is inspired by and interoperates with [ETH Zurich's Robot Learning course repository](https://github.com/mees-robot-learning-course/ethz-course-2026), specifically `hw3_imitation_learning` and its SO-101 multicube MuJoCo scene.

The course repository did not declare a repository-level license when inspected on 2 August 2026. Therefore, this repository does not copy or redistribute the course homework code, autograder, or scene assets. The smoke script loads a separately obtained local checkout. The robot mesh subdirectory in the upstream repository includes its own license; users must still review upstream terms before redistribution.

All code authored in this repository is released under the MIT License.
