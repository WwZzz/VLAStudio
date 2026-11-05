# Rebot B601-RS on Jetson

Implemented in `/home/jetson/Project/VLAStudio`, branch `dev_update`.
The adapter uses MotorBridge directly and the existing VLAStudio device/SHM
interfaces. LeRobot itself is not required. The Alicia-D hardware SDK is not used;
its relative EE, EMA and stop-hold mathematics are reused.

## Existing zero calibration

Zero calibration performed in Windows with MotorBridge Studio is motor-side and
does not need repeating just because the host changes to Jetson. If a different
tool only saved a host-side offset file, migrate that file and explicitly account
for its coordinate convention. The current adapter reads raw motor-coordinate
radians, matching the official RS Pinocchio controller. It never calls
`set_zero_position`, writes zero offsets, clears faults or performs gripper
collision calibration. LeRobot's `joint_directions` are leader-action mappings,
not additional signs to apply to the official raw-motor URDF coordinates.

Matching the physical calibration pose to the official model still matters; an
FK/IK numerical round trip alone cannot verify physical zero alignment.

## Start Quest teleoperation

On Jetson:

```bash
cd /home/jetson/Project/VLAStudio
bash scripts/rebot_b601rs_vr_tele.sh --check
bash scripts/rebot_b601rs_vr_tele.sh
```

`--check` only reads all seven motors and checks FK/IK. It never enables motors
or sends targets. `--smoke-seconds 12` runs the robot and VR service read-only.
Normal launch switches to MIT mode, enables motors and holds the measured pose;
it does not automatically home or recalibrate. Open
`https://192.168.71.118:8443` in the Quest browser (use the current Jetson IP if it
changes), accept the local certificate and enter the VR session as yesterday.

- Right-hand grip/clutch held: relative EE control. Releasing it stops accepting
  new VR poses. The output still finishes any already accepted rate-limited target.
- Right trigger: gripper travel, same mapping as the existing Alicia teleop.
- **B: return the six arm joints AND gripper motor to their existing motor zero.**
  Gripper zero is closed; B does not capture a new zero or use the startup pose.
- Ctrl+C / SIGTERM / ordinary exceptions: stop accepting VR commands, cancel the
  pending target at the last streamed command, then park in two stages (J1/4/5/6,
  followed by J2/3) with the same minimum-jerk/speed limits. Exit parking preserves
  the gripper opening instead of issuing an automatic release/closure command.
  Confirm all six joints are within 1 degree of zero and stable over three fresh
  feedback samples, then disable all seven motors and verify stopped status 0.
  B still commands all seven existing motor zeros without disabling.
- The supervisor waits for arm shutdown and does not force-kill it after 4 seconds.
  Repeated Ctrl+C/SIGTERM requests do not interrupt the cleanup trajectory.
- If feedback is stale, faults are active, parking fails, or stop feedback cannot
  be confirmed, print `SAFE EXIT FAILED` and return a failed process status. Do not
  blindly command an unverified path or torque-off at an unknown pose. Motors may
  remain enabled: support the arm and use its local stop. Socket closure is not
  proof of torque-off or indefinite holding. Firmware watchdog behavior is unchanged.
- Software cleanup cannot run after SIGKILL, power loss, or a process crash that
  prevents Python cleanup. Automatic parking has no collision/obstacle detection;
  zero feedback is not proof of mechanical support. Put held objects down and keep
  the return path clear before normal exit; disabling also removes gripper torque.

Do not run another controller on the same CAN bus at the same time. This launch
has its own session/bus locks; those do not lock out unrelated tools.

The PCAN-USB interface is `can1` at 1 Mbit/s on this Jetson; `can0` is onboard CAN.
After replugging the PCAN, check its interface name and bring it up again if needed:

```bash
ip -br link
sudo ip link set can1 type can bitrate 1000000 restart-ms 100
sudo ip link set can1 up
```

No cameras are added to this profile because none were connected during Rebot
integration. Yesterday's three-camera Alicia profile is preserved separately.

## Joint and absolute EE control

Run with the `vlastudio` conda environment and `PYTHONPATH=src`. Connect is read-only;
call `enable()` explicitly before sending targets. Do not run this example alongside
the VR process.

```python
import numpy as np
from vlastudio.deploy.robot.rebot_b601rs import RebotB601RS

robot = RebotB601RS(channel="can1")
try:
    robot.connect()
    measured = robot.get_observation()["qpos"]
    robot.enable()  # holds measured pose; no recalibration
    robot.set_joints(measured[:6], gripper=float(measured[6]))
    pose6 = robot.state_ee_from_joints()  # base-frame x,y,z,roll,pitch,yaw
    ok = robot.set_state_ee(np.r_[pose6, measured[6]])
finally:
    robot.close()
```

`publish_action`/`set_joints` use joint radians and gripper open fraction [0,1].
`set_pose` accepts a 4x4 base-frame transform; `set_state_ee` accepts [x,y,z in m,
roll,pitch,yaw in rad, optional gripper]. Failed IK sends no target.
Available robot YAMLs: `rebot_b601rs.yaml` (qpos), `rebot_b601rs_ee.yaml`
(absolute EE), `rebot_b601rs_rel_ee.yaml` (relative EE / Quest).

The reduced Pinocchio model locks both gripper slides and solves six arm joints
to the wrist/flange frame `link6` with a LOCAL Jacobian, damped least squares and
joint bounds. All three robot YAMLs explicitly set `ik_end_frame: link6`.
VR anchors, absolute EE control, FK and newly recorded `state_ee` use this same
frame. Pure rotations about a fixed wrist point can move the fingertips along an
arc. No joint zero or URDF geometry was changed.
This replaces the previous `gripper_end` TCP (166.21 mm along link6's local Z,
with an additional fixed rotation). Observations record `ee_frame` to distinguish
the convention. Old EE recordings use their original frame and must not be mixed
with wrist-frame EE values without converting the full pose. The optional
`ik_end_frame: gripper_end` remains supported for explicit legacy use.
Relative/delta VR control now uses a bounded differential IK step each input frame
instead of repeatedly searching for a full-pose solution and re-anchoring on failure.
The LOCAL SE(3) log Jacobian includes Jlog6, with damped bounded least squares and
a descent check. Joint position, per-frame speed and 0.15-rad commanded lead relative
to motor feedback remain bounded. At a singularity/unreachable target it makes a
feasible local improvement or holds; the original clutch anchor is retained until
the next clutch press. This is tracking, not a claim that every requested pose was
reached. `ik_tracking` observations report tracking/arrived/limited and residual.
Absolute EE / set_pose / set_state_ee retain the strict full-pose solver and reject
failed IK. The shared Alicia hook keeps the existing Alicia behavior unchanged.

Home confirmation keeps the 1-degree arm tolerance and now uses a separate
`home_gripper_tolerance_rad: 0.05`, greater than the 0.03-rad gripper control dead
zone. A captured failure had max arm error 0.51 degrees and gripper residual
0.03693 rad: the old 0.03-rad gripper check alone rejected that return. Timeout
messages now include six joint errors and gripper error/tolerance; observations
expose `home_status`. No zero calibration or torque limit was changed.
Bounds are the intersection of the official RS URDF and LeRobot RS limits.
The VR profile's arm target speed is 1 rad/s (user approved); B and shutdown parking
use a separate 0.5 rad/s backend limit and trajectory timing. General qpos/EE
profiles retain 0.5 rad/s.

The Quest profile ramps the gripper target at up to 2.5 rad/s. The former 1 rad/s
ramp required 4.71 s for full travel even before allowing for motor lag.
Motor-side damping blends from 1.5 to 0.40 only when fresh feedback shows motion
toward a distant target. It returns to 1.5 on a stall, reversal, or approach
within 0.25 rad of the target. A feedback interval or age above 120 ms inhibits
fast travel until fresh motion is confirmed. A blocked grip can accumulate at
most 0.164 rad of command lead at the default gains. Reversing the trigger
across the measured position discards the old lead. The Quest trigger can also
request full closure with the wrist stationary at its anchor.

The 0.03 rad dead zone and +/-1 Nm position-error torque term apply throughout.
Motor-side velocity damping adds its own torque. Actual grip force and travel
time depend on the hardware and load. B and exit parking use the holding damping
and at most 1 rad/s for the gripper. Direct joint/EE profiles use the original
1 rad/s ramp and fixed 1.5 damping by default. Startup uses a bounded hold at the
measured position. Observations expose raw requested/ramped/measured position,
measured speed, sample age, applied damping and drive torque in `gripper_control`.
VR input requests 60 Hz; joint EMA alpha is 0.5 with 10 still frames. Shared stop-hold
logic waits for EMA convergence and resumes on cumulative motion from the held pose.
The bus worker requests 100 Hz and receives independent active reports from motor
IDs 1–7. Each joint carries its SocketCAN arrival time in the host monotonic clock.
At connection and before enabling, each reported position must agree with its
mechanical position register (0x7019) within 1 degree while the arm is stationary.
Disconnect restores the reporting streams enabled by this session. A joint whose
latest report is older than 0.3 s is identified by ID. Stale feedback or a reported
motor fault stops the worker and rejects further targets. Rebot does not inherit
the earlier Alicia E1/E2 filtering. This integration does not modify firmware fault
or watchdog behavior and does not guarantee torque-off when the worker stops.

Saved observation `qpos` is measured feedback, never a copy of the target.
`motor_positions` includes all seven raw radians. The dataset hook computes
`state_ee` using the Rebot geometry; `obs2meta` exposes state/state_joint/state_ee.

## Dependencies and sources

Verified runtime: Python 3.10, MotorBridge 0.5.6, Pinocchio 2.7.0, NumPy 1.26.4.
See `requirements/rebot_b601rs_jetson.txt`; the existing JetPack/PyTorch runtime
is preserved without a NumPy major-version upgrade.

- Official Pinocchio controller and bundled URDF:
  https://github.com/Seeed-Projects/reBotArm_control_py
  commit `d4f3591bc0183de03f3ae8651298db0269aadfdd`.
  `assets/ReBot_Arm_RS.urdf` is copied from that source with its existing header.
  Mesh assets are not bundled; this profile uses FK/IK without visualization.
- Official LeRobot RS motor IDs, models, limits and MIT gains:
  https://github.com/Seeed-Projects/lerobot-robot-seeed-b601
  commit `0da1a1a5575905d77823227003ffee7672059780`.
- https://wiki.seeedstudio.com/rebot_arm_b601_rs_pinocchio_meshcat/
- https://wiki.seeedstudio.com/rebot_b601_rs_getting_started/

## Verification and limitation (2026-11-05)

Seven motor IDs responded on can1, with zero firmware fault reports during the
initial reads. Real read-only FK/IK and a 12-second native VR startup passed.
The mock tests cover coordinate units, home semantics, failed IK rejection,
recording geometry, read-only transport, staged parking before disable, preserving
gripper opening, failed/stale parking rejection, repeated exit requests, no output
after disable, attempting all seven disables after one fails, and supervisor waits.
On the user's explicit request all seven real motors were disabled and returned
stopped status 0. Subsequent tests do not re-enable the real arm. Actual VR following, physical EE
accuracy and a full-range B return have not been visually verified.
The updated exit implementation passed 20 isolated tests plus a real read-only
VR service startup/shutdown. The powered parking/disable sequence is covered with
simulated motors; it has not been exercised by moving the real arm during this change.

Latency diagnosis: pure IK on nearby reachable targets measured approximately
0.39 ms median for Rebot and 11.7 ms for both Alicia arms combined. The previous read-only CAN
query loop measured about 99 Hz, position queries 8.13 ms median, fault reads
0.019 ms; oldest joint feedback age median 64 ms. Powered MIT output/physical
tracking and Quest network delay were not measured by this test.
The previous EMA (0.15 at 30 Hz) has ~189 ms theoretical low-frequency delay; the
new 0.5 at requested 60 Hz has ~17 ms if new actions are actually processed at that
rate (~33 ms at 30 Hz). These are filter calculations, not end-to-end measurements.
Run `PYTHONPATH=src /home/jetson/miniforge3/envs/vlastudio/bin/python
scripts/benchmark_teleop_latency.py --read-can` for query-only timing. It never
enables motors or publishes targets.

An initial unit-test injection bug allowed tests to connect to the real bus and
send enable/home targets while the arm was already reporting near zero. No zero
calibration was written. The injection scope was corrected and a separate guard
now forbids real CAN connections in robot unit tests. Subsequent tests use mocks;
the native VR startup check uses read-only transport.
