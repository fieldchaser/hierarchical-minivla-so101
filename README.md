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
- [x] Record or generate MuJoCo multicube demonstrations
- [x] Train a neural state + one-hot goal baseline on real simulation data
- [ ] Add RGB observations and natural-language goals (Flat MiniVLA)
- [ ] Add reach/grasp/transport/release supervision (Hierarchical MiniVLA)
- [ ] Evaluate unseen layouts and instruction paraphrases

## Upstream attribution and licensing boundary

This project is inspired by and interoperates with [ETH Zurich's Robot Learning course repository](https://github.com/mees-robot-learning-course/ethz-course-2026), specifically `hw3_imitation_learning` and its SO-101 multicube MuJoCo scene.

The course repository did not declare a repository-level license when inspected on 2 August 2026. Therefore, this repository does not copy or redistribute the course homework code, autograder, or scene assets. The smoke script loads a separately obtained local checkout. The robot mesh subdirectory in the upstream repository includes its own license; users must still review upstream terms before redistribution.

All code authored in this repository is released under the MIT License.

## Milestone 1: physically executed scripted expert

The first real MuJoCo demonstration now uses a deterministic finite-state
controller instead of synthetic labels:

```text
reach -> descend -> grasp -> lift -> transport -> release
```

The expert controls the existing mocap end-effector target and the physical jaw
actuator in ETH's fixed multicube scene. The cube is never attached or moved by
editing its state. During a successful grasp, the jaw stops near `0.32 rad`
instead of reaching its closed limit, showing that the cube is held by contact.
Success additionally requires the jaw to be open after the cube settles in the
bin.

Run the collector with a separately obtained ETH HW3 checkout:

```bash
python scripts/collect_scripted_demo.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning \
  --output data/scripted/red_fixed_seed0.npz
```

The compressed trajectory contains aligned observations and commands:

```text
state_ee_xyz, state_joints, state_gripper
cubes_xyz, cubes, state_goal, goal_pos
action_ee_xyz, action_gripper
phase, phase_names
```

In the verified fixed-layout episode, the red cube was physically grasped,
lifted, transported, released, and remained inside the bin after settling. This
milestone proves the classical-control-to-demonstration part of the pipeline;
randomized layouts and learned policies remain future work.

Verification was repeated three times from a fresh reset. Every run succeeded
in 273 control steps with a grasp contact angle of `0.3255 rad`; the final cube
center was `[-0.1959, 0.6890, 0.0233]` for a bin centered at
`[-0.2000, 0.7000, 0.0210]`. All ten temporal arrays in each saved trajectory
have the same length, and the complete project test suite contains eight passing
tests.

## Milestone 2: randomized multi-goal dataset

The same physical expert now runs on shuffled layouts with Gaussian position
noise and cycles through red, green, and blue goal conditions. A batch
collector saves only successful trajectories for behavior cloning and writes a
`manifest.json` containing every attempted seed, including final cube and bin
positions for failures.

Collect the default 20-episode validation set with:

```bash
python scripts/collect_scripted_dataset.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning \
  --output-dir data/scripted/randomized \
  --episodes 20
```

The command exits unsuccessfully when fewer than 90% of attempts work, so it can
also serve as a reproducible regression check. Failed attempts are documented in
the manifest but are not added to the training trajectories.

On 2 August 2026, seeds 0 through 19 achieved 19/20 successful physical
pick-and-place episodes (95%) and produced 4,203 aligned state-action pairs
while cycling through all three goal colors. Seed
18 was the only failure: the red cube remained near the gripper at a height of
approximately 8.2 cm after the release phase. This gives us a real randomized
MuJoCo dataset for the next milestone: a learned neural state-and-goal policy.

## Milestone 3: learned state-and-goal baseline

The first learned MuJoCo policy is a two-layer MLP. It receives robot state,
all cube positions, the one-hot task goal, and the bin position. As in the
linear smoke baseline, the input also makes goal selection explicit by adding
the selected cube position and relative target/bin geometry. The policy
regresses a bounded Cartesian delta and classifies the gripper as open or
closed.

Install the learning dependency, train, and evaluate with:

```bash
python -m pip install -e '.[learn,sim]'

python scripts/train_state_policy.py \
  --data-dir data/scripted/randomized \
  --output checkpoints/state_goal_policy.pt \
  --metrics-output results/state_policy_offline.json

python scripts/evaluate_state_policy.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning \
  --checkpoint checkpoints/state_goal_policy.pt \
  --episodes 10 \
  --seed-start 100 \
  --output results/state_policy_unseen.json
```

The split is performed by whole episode, rather than by randomly mixing adjacent
timesteps. Fifteen episodes (3,248 steps) were used for training and four
episodes (955 steps) for validation. Early stopping selected epoch 31.

| Metric | Result |
|---|---:|
| Validation Cartesian MAE | 0.766 mm |
| Validation gripper accuracy | 98.95% |
| Closed loop, collected seeds 0-9 | 0/10 |
| Closed loop, unseen seeds 100-109 | 0/10 |

The zero closed-loop success rate is retained as a negative baseline, not hidden
behind the strong offline metrics. The learned controller frequently grasps and
lifts the correct cube, but does not reliably transition from lifting to
transport and release. Small one-step errors compound until the state leaves the
demonstration distribution. A conservative workspace bound prevents these
errors from driving the MuJoCo mocap target into numerically unsafe regions.

This failure identifies the next concrete research question: whether explicit
phase supervision can resolve the long-horizon ambiguity that the flat policy
cannot. The existing demonstrations already contain reach, descend, grasp,
lift, transport, and release labels, so the next comparison can hold the data
and network size fixed while changing only the policy structure. Exact metrics
and per-seed outcomes are stored under `results/`.
