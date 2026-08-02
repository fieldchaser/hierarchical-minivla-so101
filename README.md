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
- [x] Run DAgger round 2 from the updated policy
- [x] Target the descend-to-grasp boundary in DAgger round 3
- [x] Extend on-policy aggregation through grasp and lift
- [x] Aggregate on-policy transport and alignment failures
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

## Milestone 7: on-policy boundary DAgger, round 2

Round 2 rolls out the round-1 checkpoint on 30 new randomized scenes. This is
the defining DAgger loop: each new dataset follows the updated learner's state
distribution instead of replaying failures from the original policy.

```bash
python scripts/collect_dagger_dataset.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning \
  --checkpoint checkpoints/hierarchical_state_policy_dagger_boundary_r1.pt \
  --output-dir data/dagger/round_2_boundary \
  --episodes 30 \
  --seed-start 500

python scripts/train_hierarchical_state_policy.py \
  --data-dir data/scripted/recovery_mocap_120 \
             data/dagger/round_1_boundary \
             data/dagger/round_2_boundary \
  --output checkpoints/hierarchical_state_policy_dagger_boundary_r2.pt \
  --metrics-output results/hierarchical_state_dagger_boundary_r2_offline.json \
  --epochs 300
```

The hybrid learner/expert collector succeeded in 28/30 scenes (93.33%), up
from 25/30 in round 1. It saved 4,509 oracle-labeled states, including 1,350
states reached under learner control. Aggregating all successful scripted and
DAgger episodes produced 36,125 training and validation steps across 173
episodes.

| Metric | Boundary DAgger R1 | Boundary DAgger R2 |
|---|---:|---:|
| Hybrid collection success | 25/30 | 28/30 |
| Learner-generated labeled states | 1,097 | 1,350 |
| Validation Cartesian MAE | 0.263 mm | 0.270 mm |
| Validation gripper accuracy | 100% | 100% |
| Validation phase accuracy | 91.95% | 93.02% |
| Full learned-policy success | 0/10 | 0/10 |
| Reached descend, same unseen seeds 600-609 | 4/10 | 7/10 |

Because task success is a sparse terminal metric, evaluation now also records
the terminal phase distribution, mean final phase, and how many rollouts
reached each phase. On the same unseen layouts, round 2 raises mean final phase
from `0.4` to `0.7` and moves three additional rollouts from reach into descend.
This is genuine intermediate progress, but not task completion: no rollout
reaches grasp, and final success remains 0/10.

The current bottleneck has therefore shifted from the reach-to-descend boundary
to descend-to-grasp. The next DAgger round should use the round-2 checkpoint and
collect new on-policy corrections around that boundary before any claim that
the state controller is solved. RGB and language remain intentionally deferred
until the closed-loop control pipeline has a stronger base.

## Milestone 8: descend-to-grasp DAgger, round 3

Round 3 rolls out the round-2 checkpoint on seeds 700-729 and aggregates the
new state distribution with every previous successful episode:

```bash
python scripts/collect_dagger_dataset.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning \
  --checkpoint checkpoints/hierarchical_state_policy_dagger_boundary_r2.pt \
  --output-dir data/dagger/round_3_descend_grasp \
  --episodes 30 \
  --seed-start 700

python scripts/train_hierarchical_state_policy.py \
  --data-dir data/scripted/recovery_mocap_120 \
             data/dagger/round_1_boundary \
             data/dagger/round_2_boundary \
             data/dagger/round_3_descend_grasp \
  --output checkpoints/hierarchical_state_policy_dagger_descend_grasp_r3.pt \
  --metrics-output results/hierarchical_state_dagger_descend_grasp_r3_offline.json \
  --epochs 300
```

The hybrid collector succeeded in 27/30 attempts (90%) and saved 4,023
oracle-labeled states, 1,304 of which were reached under learner control. The
four aggregated datasets contain 40,148 steps from 200 successful episodes.

| Metric | Boundary DAgger R2 | Descend/grasp DAgger R3 |
|---|---:|---:|
| Validation Cartesian MAE | 0.270 mm | 0.260 mm |
| Validation gripper accuracy | 100% | 100% |
| Validation phase accuracy | 93.02% | 93.31% |
| Reached descend, same unseen seeds 800-809 | 7/10 | 5/10 |
| Reached grasp with learned transitions | 0/10 | 0/10 |
| Full learned-policy success | 0/10 | 0/10 |

This is a useful negative result: a third aggregation round improves offline
imitation metrics but does not fix the closed-loop boundary. Terminal-state
diagnostics show that all ten rollouts stop within 4 mm of their active reach or
descend target, with nearly zero predicted motion, while the phase head keeps
predicting the current phase. The training label switches only after the expert
crosses this narrow boundary, so collecting more current-phase corrections
cannot reliably teach a completion event.

The evaluator therefore adds an explicit `observed` event diagnostic. It uses
only the mocap, target-cube, bin, and gripper values already present in the
state-policy observation; it does not move objects or provide expert actions:

```bash
python scripts/evaluate_hierarchical_state_policy.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning \
  --checkpoint checkpoints/hierarchical_state_policy_dagger_descend_grasp_r3.pt \
  --episodes 10 \
  --seed-start 800 \
  --phase-source observed \
  --output results/hierarchical_state_dagger_descend_grasp_r3_observed_800.json
```

With observed completion events, all 10/10 rollouts reach grasp and lift, and
2/10 reach transport and release. Full success remains 0/10 because most cubes
are lost during lift and the two release trajectories do not settle in the bin.
This separates the two remaining problems: narrow learned phase boundaries and
insufficient on-policy grasp/lift recovery data. The observed mode is a
state-policy diagnostic, not a claim of RGB-only autonomy; a later VLA must
learn equivalent completion events from visual and proprioceptive inputs.

The next milestone will keep these event gates and extend learner execution plus
expert correction into grasp and lift. Repeating another reach/descend-only
DAgger round is not justified by the round-3 evidence.

## Milestone 9: grasp/lift DAgger, round 4

Round 4 extends learner execution into the contact-sensitive grasp and lift
phases. The approach phases retain an 80-step learner budget, while grasp and
lift use a conservative 20-step budget before the scripted expert takes over.
Observed completion events select the action head during collection, avoiding
the narrow learned phase-boundary failure diagnosed in round 3.

```bash
python scripts/collect_dagger_dataset.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning \
  --checkpoint checkpoints/hierarchical_state_policy_dagger_descend_grasp_r3.pt \
  --output-dir data/dagger/round_4_grasp_lift \
  --episodes 30 \
  --seed-start 900 \
  --learner-grasp-lift-steps 20 \
  --observed-phase-events
```

A six-episode smoke run first verified that the new configuration really visits
the intended phases. Formal collection then succeeded in 28/30 attempts
(93.33%) and saved 4,811 oracle-labeled states. Of the 2,596 learner-generated
states, 560 are grasp states and 560 are lift states; previous rounds contained
no learner execution in either phase.

Training aggregates all five datasets:

```bash
python scripts/train_hierarchical_state_policy.py \
  --data-dir data/scripted/recovery_mocap_120 \
             data/dagger/round_1_boundary \
             data/dagger/round_2_boundary \
             data/dagger/round_3_descend_grasp \
             data/dagger/round_4_grasp_lift \
  --output checkpoints/hierarchical_state_policy_dagger_grasp_lift_r4.pt \
  --metrics-output results/hierarchical_state_dagger_grasp_lift_r4_offline.json \
  --epochs 300
```

The resulting split contains 44,959 steps from 228 successful episodes. The
best checkpoint was selected at epoch 98.

| Metric | Descend/grasp R3 | Grasp/lift R4 |
|---|---:|---:|
| Validation Cartesian MAE | 0.260 mm | 0.270 mm |
| Validation gripper accuracy | 100% | 100% |
| Validation phase accuracy | 93.31% | 94.47% |
| Grasp phase accuracy | 96.67% | 97.01% |
| Lift phase accuracy | 81.21% | 87.60% |
| Success, same unseen seeds 1000-1009 | 8/10 | 10/10 |

Before the final comparison, the observed event gates were aligned exactly with
the data-collection protocol. Both now require a 4 mm approach tolerance, 30
grasp-settling steps, and 12 mm object-to-bin transport tolerance. The earlier
diagnostic used looser 12 mm, 10 mm, and 50 mm gates, which caused premature
grasp and release transitions. Shared constants and boundary tests now prevent
the collection and evaluation contracts from drifting apart again.

With the aligned protocol, round 4 achieves 10/10 success on collected-layout
seeds 0-9 and 10/10 on the held-out comparison seeds 1000-1009. A larger test on
30 new seeds 1100-1129 succeeds in 22/30 episodes (73.33%). All 30 episodes
complete grasp and lift and enter transport; the eight failures all remain in
transport without satisfying the 12 mm release condition.

This is the first learned-action checkpoint in the project to complete the full
pick-and-place loop. The event gates use state observations to decide when a
skill is complete, but every Cartesian and gripper command during evaluation is
produced by the learned phase-specific action heads; the expert does not take
over. It remains a hierarchical state-policy result rather than a VLA result.

The next milestone will extend on-policy expert correction into transport and
focus on object-centered bin alignment. The target is to reduce the eight
transport failures before replacing state observations with RGB and language.

## Milestone 10: transport DAgger, round 5

Round 5 adds a separate transport learner budget while preserving the default
behavior of every previous collection command. With a 20-step budget, the
learner begins moving the held cube toward the bin before the expert takes over
and labels the remaining object-centered alignment correction.

```bash
python scripts/collect_dagger_dataset.py \
  --eth-hw3 /path/to/ethz-course-2026/hw3_imitation_learning \
  --checkpoint checkpoints/hierarchical_state_policy_dagger_grasp_lift_r4.pt \
  --output-dir data/dagger/round_5_transport \
  --episodes 30 \
  --seed-start 1200 \
  --learner-grasp-lift-steps 20 \
  --learner-transport-steps 20 \
  --observed-phase-events
```

The hybrid collector succeeded in 26/30 attempts (86.67%) and saved 4,522
expert-labeled states. It contains 2,857 learner-generated states, including
520 transport states; all earlier rounds had zero learner execution in
transport.

```bash
python scripts/train_hierarchical_state_policy.py \
  --data-dir data/scripted/recovery_mocap_120 \
             data/dagger/round_1_boundary \
             data/dagger/round_2_boundary \
             data/dagger/round_3_descend_grasp \
             data/dagger/round_4_grasp_lift \
             data/dagger/round_5_transport \
  --output checkpoints/hierarchical_state_policy_dagger_transport_r5.pt \
  --metrics-output results/hierarchical_state_dagger_transport_r5_offline.json \
  --epochs 300
```

The six aggregated datasets contain 49,481 steps from 254 successful episodes.
Training selected epoch 66, with 94.97% validation phase accuracy. Overall
Cartesian MAE increased from 0.270 mm to 0.336 mm because the new aggregation
adds harder off-trajectory states.

On the round-5 dataset itself, the new checkpoint lowers transport action MAE
from 0.453 mm to 0.326 mm and lift MAE from 1.438 mm to 0.929 mm. That local
improvement does not translate into a better full policy:

| Closed-loop comparison | Grasp/lift R4 | Transport R5 |
|---|---:|---:|
| Same unseen seeds 1300-1309 | 9/10 | 7/10 |
| Same unseen seeds 1400-1429 | 24/30 | 22/30 |
| Final descend failures, seeds 1400-1429 | 0 | 1 |
| Final lift failures, seeds 1400-1429 | 2 | 1 |
| Final transport failures, seeds 1400-1429 | 4 | 6 |

Round 5 is therefore retained as a negative DAgger result rather than promoted
as the new baseline. Transport corrections improve one-step imitation on their
own state distribution, but retraining the full shared encoder slightly harms
the end-to-end state distribution. This illustrates why offline action error
alone is not a policy-selection metric for long-horizon control.

The round-4 checkpoint remains the recommended state-policy baseline. Further
state-only DAgger rounds are deferred: the project now has a reproducible
hierarchical controller with 80% success on a 30-seed held-out comparison, and
the next milestone moves to RGB observations and language goals. Round 5 also
provides a concrete future ablation for transport-head-only fine-tuning without
changing the shared encoder.
