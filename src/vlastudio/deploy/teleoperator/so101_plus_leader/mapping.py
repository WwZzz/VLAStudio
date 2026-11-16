"""Piecewise linear calibration in measured leader range units and follower qpos."""
import numpy as np

# Rebot J5 is yaw, J6 is roll. Servo ID order differs from kinematic order.
ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
         "wrist_yaw", "wrist_roll", "gripper")


class JointMap:
    @classmethod
    def from_reference(cls, joints, ref):
        rows = dict(joints)
        rows["gripper"] = dict(rows["gripper"], leader=[ref["closed"], ref["opened"]])
        return cls(rows, ref["joints"])

    def __init__(self, joints, reference):
        if set(joints) != set(ORDER):
            raise ValueError("Configure all seven leader joints by name")
        self.axes = []
        for name in ORDER:
            row = joints[name]
            home = reference[name]
            points = row["leader"]
            far = -100 if home > 0 else 100
            x = np.array([{"home": home, "far": far}.get(v, v) for v in points], dtype=float)
            y = np.array(row["target"], dtype=float)
            if name != "gripper":
                y = np.deg2rad(y)
            if x.shape != y.shape or x.ndim != 1 or len(x) < 2 or not np.isfinite([x, y]).all():
                raise ValueError(f"Invalid landmarks for {name}")
            if np.all(np.diff(x) < 0):
                x, y = x[::-1], y[::-1]
            if (not np.all(np.diff(x) > 0) or not (np.all(np.diff(y)>0) or np.all(np.diff(y)<0))
                    or x[0] < (0 if name == "gripper" else -100) or x[-1] > 100):
                raise ValueError(f"Invalid ordered landmarks for {name}: {x}, {y}")
            clip = row.get("clip", True)
            if not isinstance(clip, bool):
                raise ValueError(f"{name}.clip must be boolean")
            self.axes.append((x, y, clip))

    def target(self, measured, previous):
        result = []
        for i, (x, y, clip) in enumerate(self.axes):
            v = measured[i]
            result.append(np.interp(v, x, y) if clip or x[0]<=v<=x[-1] else previous[i])
        return np.asarray(result)

    def leader(self, target):
        if np.shape(target) != (7,) or not np.isfinite(target).all():
            raise ValueError("Expected seven finite follower coordinates")
        result = []
        for q, (x, y, _) in zip(target, self.axes):
            if y[0] > y[-1]:
                x, y = x[::-1], y[::-1]
            # Encoder noise at a joint stop is expected.
            if not y[0]-.018 <= q <= y[-1]+.018:
                raise ValueError("Follower pose is outside configured leader mapping")
            result.append(np.interp(q,y,x))
        return np.asarray(result)
