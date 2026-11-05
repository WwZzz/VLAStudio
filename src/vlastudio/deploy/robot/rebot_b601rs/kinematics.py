"""Pinocchio LOCAL-Jacobian damped IK, following Seeed's RS CLIK approach."""
import time
import numpy as np
import pinocchio as pin
from scipy.optimize import lsq_linear


class RebotIK:
    def __init__(self, urdf, end_frame="link6", max_iter=80, tolerance=.001,
                 budget_s=.022):
        full = pin.buildModelFromUrdf(str(urdf))
        locks = [full.getJointId(n) for n in ("gripper_joint1", "gripper_joint2")]
        self.reduced_model = pin.buildReducedModel(full, locks, pin.neutral(full))
        assert self.reduced_model.nq == self.reduced_model.nv == 6
        assert list(self.reduced_model.names)[1:] == [f"joint{i}" for i in range(1, 7)]
        self.reduced_data = self.reduced_model.createData()
        self.end_frame = end_frame
        self.frame = self.reduced_model.getFrameId(end_frame)
        if self.frame >= self.reduced_model.nframes:
            raise ValueError("Unknown end frame: " + end_frame)
        # Intersection of official URDF and LeRobot's RS motor-coordinate limits.
        self.lower = np.maximum(self.reduced_model.lowerPositionLimit,
                                np.deg2rad([-145, 0, 0, -80, -90, -130]))
        self.upper = np.minimum(self.reduced_model.upperPositionLimit,
                                np.deg2rad([145, 170, 200, 90, 90, 130]))
        self.max_iter, self.tol, self.budget = max_iter, tolerance, budget_s
        self.init_data = np.zeros(6)
        self._last_solve_timed_out = False

    def reset(self, q=None):
        self.init_data = np.asarray(np.zeros(6) if q is None else q, dtype=float).copy()

    def set_reference(self, q, pose=None):
        self.reset(q)

    def compute_fk(self, q):
        q = np.asarray(q, dtype=float)
        pin.forwardKinematics(self.reduced_model, self.reduced_data, q)
        pin.updateFramePlacements(self.reduced_model, self.reduced_data)
        return self.reduced_data.oMf[self.frame].homogeneous.copy()

    def get_current_pose(self):
        return self.compute_fk(self.init_data)

    def track(self, target_pose, q_reference, max_step, measured=None, max_lead=.15):
        """One bounded differential-IK update; does not claim exact pose arrival.

        No branch search or warm-start reset. At a singularity/limit, return the
        locally feasible improvement (or hold), retaining the original EE target.
        Strict absolute-pose callers continue to use solve().
        """
        q=np.clip(np.asarray(q_reference,dtype=float),self.lower,self.upper)
        target=pin.SE3(np.asarray(target_pose,dtype=float))
        current=pin.SE3(self.compute_fk(q))
        relative=current.inverse()*target
        error=pin.log6(relative).vector
        before=float(np.linalg.norm(error))
        if before<=self.tol:
            return q.copy(),dict(status='arrived',residual=before,step_rad=0.)
        pin.computeJointJacobians(self.reduced_model,self.reduced_data,q)
        jac=pin.getFrameJacobian(self.reduced_model,self.reduced_data,self.frame,pin.LOCAL)
        jac=pin.Jlog6(relative.inverse())@jac
        step=float(max_step)
        lower=np.maximum(self.lower-q,-step)
        upper=np.minimum(self.upper-q,step)
        if measured is not None:
            delta=np.asarray(measured)-q
            # Current reference remains feasible even when feedback already lags:
            # allow recovery toward feedback, but no additional outward lead.
            lower=np.maximum(lower,np.minimum(0.,delta-max_lead))
            upper=np.minimum(upper,np.maximum(0.,delta+max_lead))
        free=(upper-lower)>1e-10
        dq=np.zeros(6)
        if free.any():
            # Damping makes the least-squares problem well conditioned near wrist
            # singularities; box bounds limit the TOTAL update, not each iteration.
            A=np.vstack([jac[:,free],np.eye(6)[:,free]*.01])
            b=np.r_[error,np.zeros(6)]
            result=lsq_linear(A,b,bounds=(lower[free],upper[free]),method='bvls',max_iter=30,tol=1e-8)
            if np.isfinite(result.x).all():
                dq[free]=result.x
        for scale in (1.,.5,.25,.125):
            candidate=np.clip(q+scale*dq,self.lower,self.upper)
            residual=float(np.linalg.norm(pin.log6(pin.SE3(self.compute_fk(candidate)).inverse()*target).vector))
            if residual<before-1e-10:
                self.init_data=candidate.copy()
                return candidate,dict(status='tracking',residual=residual,
                                      step_rad=float(max(abs(candidate-q))))
        self.init_data=q.copy()
        return q.copy(),dict(status='limited',residual=before,step_rad=0.)

    def solve(self, target_pose, q_init=None):
        target = pin.SE3(np.asarray(target_pose))
        q = np.clip(self.init_data if q_init is None else q_init, self.lower, self.upper).copy()
        begin = time.monotonic()
        self._last_solve_timed_out = False
        error = float('inf')
        for iteration in range(self.max_iter):
            current = pin.SE3(self.compute_fk(q))
            e = pin.log6(current.inverse() * target).vector
            error = float(np.linalg.norm(e))
            if error <= self.tol:
                self.init_data = q.copy()
                return True, q, iteration, error
            if self.budget and time.monotonic() - begin > self.budget:
                self._last_solve_timed_out = True
                break
            pin.computeJointJacobians(self.reduced_model, self.reduced_data, q)
            J = pin.getFrameJacobian(self.reduced_model, self.reduced_data, self.frame, pin.LOCAL)
            # Re-solve without joints trying to move outward at an active limit.
            free = np.ones(6, dtype=bool)
            dq = np.zeros(6)
            for _ in range(7):
                jf = J[:, free]
                dq[:] = 0
                dq[free] = jf.T @ np.linalg.solve(jf @ jf.T + np.eye(6)*1e-5, e)
                blocked = ((q <= self.lower+1e-9) & (dq < 0)) | ((q >= self.upper-1e-9) & (dq > 0))
                blocked &= free
                if not blocked.any():
                    break
                free[blocked] = False
            dq *= min(1., .1 / max(float(np.max(np.abs(dq))), 1e-9))
            for scale in (1., .5, .25, .125, .0625):
                candidate = np.clip(pin.integrate(self.reduced_model, q, scale*dq), self.lower, self.upper)
                trial = pin.log6(pin.SE3(self.compute_fk(candidate)).inverse() * target).vector
                if np.linalg.norm(trial) < error:
                    q = candidate
                    break
            else:
                break
        return False, q, iteration+1, error
