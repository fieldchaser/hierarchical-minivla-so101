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
- [x] Compare flat and hierarchical state policies
- [x] Expand to recovery-augmented 120-episode data
- [x] Aggregate on-policy DAgger corrections (round 1)
- [ ] Run DAgger round 2 from the updated policy
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

## Milestone 4: hierarchical state-policy diagnostic

The hierarchical policy keeps the same 34-dimensional state-and-goal input and
two-layer 128-unit encoder. It adds a weighted six-class phase head and six
phase-specific action heads:

```text
state + one-hot goal -> shared encoder -> phase prediction
                                    \-> phase-specific action
```

At deployment, a monotonic tracker requires repeated evidence for exactly the
next phase, so the learned controller cannot move backward or skip manipulation
stages. Training and evaluation commands are:

```bash
python scripts/train_hierarchical_state_policy.py \
  --data-dir data/scripted/randomized \
  --output checkpoints/hierarchical_state_policy.pt \
  --metrics-output results/hierarchical_state_offline.json

python scripts/evaluate_hierarchical_state_policy.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning \
  --checkpoint checkpoints/hierarchical_state_policy.pt \
  --episodes 10 \
  --seed-start 100 \
  --output results/hierarchical_state_unseen.json
```

Using the identical 15/4 episode split, hierarchy improved every offline action
metric but did not yet produce a successful closed-loop episode:

| Metric | Flat | Hierarchical |
|---|---:|---:|
| Validation Cartesian MAE | 0.766 mm | 0.457 mm |
| Validation gripper accuracy | 98.95% | 100% |
| Validation phase accuracy | — | 81.68% |
| Collected-layout closed loop | 0/10 | 0/10 |
| Unseen-layout closed loop | 0/10 | 0/10 |

The weakest phase classes were lift (61.29%) and transport (74.90%). An
additional diagnostic supplied phase transitions from privileged simulator
geometry while retaining the learned action heads. This oracle-phase condition
also scored 0/10 on both layout sets. It is not a deployable policy; it isolates
the failure source by showing that phase classification alone is not the main
bottleneck.

The result supports a narrower next step: collect broader state coverage and
DAgger-style recovery corrections before adding images. Both flat and
hierarchical one-step behavior cloning currently fit expert trajectories but do
not recover after their own small errors move them away from those trajectories.
The complete learned-phase and oracle-phase outcomes are stored under
`results/`.

## Milestone 5: recovery augmentation and state-contract correction

The collector can now inject small, reproducible Cartesian perturbations after
the expert reaches the safe reach, lift, and transport targets. It does not edit
cube state or treat the injected motion as an expert action. Only the physical
controller's subsequent correction is added to the behavior-cloning data, and
every corrective row is marked by a `recovery` flag.

```bash
python scripts/collect_scripted_dataset.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning \
  --output-dir data/scripted/recovery_mocap_120 \
  --episodes 120 \
  --recovery-pos-std 0.006
```

All 120 attempts succeeded, evenly covering the three goal colors. The dataset
contains 27,631 aligned state-action pairs, including 768 explicitly perturbed
recovery rows. Data remains local and ignored by Git; the manifest records the
generation settings and every attempted seed.

This milestone also corrected an important state-action contract mismatch. The
expert computes Cartesian commands relative to the MuJoCo mocap target, while
the original learner observed only the lagging physical end-effector position.
New trajectories store `state_mocap_xyz`, increasing the learned state input
from 34 to 37 dimensions. Evaluation remains backward compatible with old
34-dimensional checkpoints.

Retraining on the corrected 120-episode dataset produced:

| Metric | 19-episode Flat | 120-episode Flat | 19-episode Hier. | 120-episode Hier. |
|---|---:|---:|---:|---:|
| Cartesian MAE | 0.766 mm | 0.421 mm | 0.457 mm | 0.277 mm |
| Gripper accuracy | 98.95% | 99.11% | 100% | 100% |
| Phase accuracy | — | — | 81.68% | 94.51% |
| Collected-layout success | 0/10 | 0/10 | 0/10 | 0/10 |
| Unseen-layout success | 0/10 | 0/10 | 0/10 | 0/10 |

The larger corrected dataset therefore improves imitation quality but still
does not solve closed-loop control. A representative flat rollout converges to
the descend/grasp height and outputs zero motion because one state must represent
multiple stages. The hierarchical reach head converges about 6 mm from its
phase boundary, where both its action and predicted phase remain at reach.
Oracle phase transitions move several episodes through release, but the cube
still stops near the bin at roughly 8 cm height rather than settling inside it.

Random perturbations around expert waypoints are useful augmentation, but they
are not yet true DAgger because the learned policy does not determine which
states receive labels. The next milestone must roll out the current policy,
capture its actual boundary failures, and query the scripted expert for recovery
actions at those states. That on-policy aggregation step is now required before
adding image observations.

## Milestone 6: on-policy boundary DAgger, round 1

The DAgger collector alternates learned control with a scripted recovery oracle.
At every visited state it stores the oracle action and phase as the training
label, together with the action that was actually executed. The first round
targets the dominant reach/descend failure: the learner runs until its Cartesian
action remains below `0.2 mm` for five steps (or reaches 80 steps), then the
expert takes over. Contact, lift, transport, and release remain expert-controlled
in this safety-focused round.

```bash
python scripts/collect_dagger_dataset.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning \
  --checkpoint checkpoints/hierarchical_state_policy_mocap_recovery.pt \
  --output-dir data/dagger/round_1_boundary \
  --episodes 30 \
  --seed-start 300
```

The recovery oracle also gained two physically motivated corrections discovered
during smoke testing: transport continuously recomputes the wrist-to-cube offset
so the cube itself is centered over the bin, and release opens the gripper while
retreating upward. With these corrections, 25/30 hybrid episodes succeeded
(83.33%). The saved set contains 3,985 labeled states, including 1,097 states
actually produced by the learner running to a phase boundary.

The training command now accepts multiple data directories, allowing aggregation
without copying or rewriting prior episodes:

```bash
python scripts/train_hierarchical_state_policy.py \
  --data-dir data/scripted/recovery_mocap_120 data/dagger/round_1_boundary \
  --output checkpoints/hierarchical_state_policy_dagger_boundary_r1.pt \
  --metrics-output results/hierarchical_state_dagger_boundary_r1_offline.json
```

| Metric | Before DAgger | Boundary DAgger R1 |
|---|---:|---:|
| Validation Cartesian MAE | 0.277 mm | 0.263 mm |
| Validation gripper accuracy | 100% | 100% |
| Validation phase accuracy | 94.51% | 91.95% |
| Collected-layout success | 0/10 | 0/10 |
| Unseen seeds 400-409 | — | 0/10 |

The lower phase accuracy is expected because the aggregated data contains more
ambiguous boundary states. Round 1 does not yet improve final task success,
although several rollouts now advance into descend. This is retained as a
negative result: DAgger is iterative, and a single round labels failures from
the pre-DAgger policy only. Round 2 must roll out the updated checkpoint so that
the next set reflects its new failure distribution rather than repeatedly
sampling the original one.
