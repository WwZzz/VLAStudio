"""Pure IK/filter timing, optionally query-only CAN timing. Never enables motors."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]


def stats(values):
    return dict(p50_ms=float(np.percentile(values,50)*1000),
                p95_ms=float(np.percentile(values,95)*1000),max_ms=float(max(values)*1000))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--read-can',action='store_true')
    p.add_argument('--seconds',type=float,default=4.)
    p.add_argument('--output')
    args=p.parse_args()
    from vlastudio.deploy.robot.bi_alicia_d import BiAliciaD
    from vlastudio.deploy.robot.rebot_b601rs import RebotB601RS
    report={'hardware_enable_sent':False,'hardware_motion_sent':False}
    for name,cls,config in (
        ('bi_alicia',BiAliciaD,'bi_alicia_d_rel_ee.yaml'),
        ('rebot',RebotB601RS,'rebot_b601rs_rel_ee.yaml')):
        row=yaml.safe_load((ROOT/'src/vlastudio/configs/robot'/config).read_text())[0]
        robot=cls(**row['args'])  # no connect, enable, publish, shutdown needed
        arms=[robot.left,robot.right] if name=='bi_alicia' else [robot]
        solvers=[arm.ik_solver for arm in arms]
        qs=[]
        for s in solvers:
            q=np.clip([0,.3,.4,0,.1,0],s.reduced_model.lowerPositionLimit,s.reduced_model.upperPositionLimit)
            s.reset(q)
            qs.append(q)
        samples=[]
        successes=0
        for i in range(80):
            targets=[]
            for s,q in zip(solvers,qs):
                v=q.copy();v[0]+=.03*np.sin(i*.13)
                targets.append(s.compute_fk(v))
            begin=time.perf_counter()
            for s,T in zip(solvers,targets):
                successes+=int(s.solve(T)[0])
            samples.append(time.perf_counter()-begin)
        alpha=float(row['args'].get('joint_smooth_alpha',.15))
        teleop_name='quest3_bi_alicia_rel_ee.yaml' if name=='bi_alicia' else 'quest3_rebot_b601rs_rel_ee.yaml'
        teleop=yaml.safe_load((ROOT/'src/vlastudio/configs/teleop'/teleop_name).read_text())
        input_fps=float(teleop['args']['fps'])
        report[name]=dict(ik_per_input_frame=stats(samples),ik_successes=successes,
            ik_calls=80*len(solvers),alpha=alpha,
            requested_input_fps=input_fps,
            filter_low_frequency_delay_ms=(1-alpha)/alpha/input_fps*1000)
    if args.read_can:
        from motorbridge import Controller
        from vlastudio.deploy.robot.rebot_b601rs.bus import RebotBus
        durations={'position':[],'fault':[]}
        class QueryMotor:
            def __init__(self,m):self.m=m
            def robstride_ping_host_id(self,*a):return self.m.robstride_ping_host_id(*a)
            def robstride_set_active_report(self, enabled):
                return self.m.robstride_set_active_report(enabled)
            def robstride_get_param_f32_host_id(self,*a):
                begin=time.perf_counter()
                value=self.m.robstride_get_param_f32_host_id(*a)
                durations['position'].append(time.perf_counter()-begin)
                return value
            def robstride_get_fault_report(self):
                begin=time.perf_counter()
                value=self.m.robstride_get_fault_report()
                durations['fault'].append(time.perf_counter()-begin)
                return value
            def close(self):self.m.close()
        class QueryController:
            def __init__(self,channel):self.c=Controller(channel)
            def add_robstride_motor(self,*a):return QueryMotor(self.c.add_robstride_motor(*a))
            def close_bus(self):self.c.close_bus()
            def close(self):self.c.close()
        bus=RebotBus('can1',read_only=True,controller_factory=QueryController)
        try:
            bus.connect()
            count=len(durations['fault']);ages=[];begin=time.monotonic()
            while time.monotonic()-begin<args.seconds:
                _,stamp=bus.snapshot()
                ages.append(time.monotonic()-min(stamp))
                time.sleep(.01)
            elapsed=time.monotonic()-begin
            report['readonly_can']=dict(query_calls={k:stats(v) for k,v in durations.items()},
                worker_hz=(len(durations['fault'])-count)/elapsed,
                oldest_feedback_age=stats(ages))
        finally:bus.close()
    text=json.dumps(report,indent=2)
    print(text,flush=True)
    if args.output:
        Path(args.output).write_text(text+'\n')


if __name__=='__main__':main()
