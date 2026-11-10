#!/usr/bin/env python3
"""Read-only UF850 cup-tip IK feasibility test; no motion commands."""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

J = [f'joint{i}' for i in range(1, 7)]

def xyz(raw, default='0 0 0'):
    return np.array([float(v) for v in (raw or default).split()])

def transform(x, r):
    a,b,c = r
    ca,sa,cb,sb,cc,sc = math.cos(a),math.sin(a),math.cos(b),math.sin(b),math.cos(c),math.sin(c)
    rx=np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
    ry=np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])
    rz=np.array([[cc,-sc,0],[sc,cc,0],[0,0,1]])
    T=np.eye(4);T[:3,:3]=rz@ry@rx;T[:3,3]=x
    return T

def load_chain(filename):
    joints=ET.parse(filename).getroot().findall('joint')
    by_child={j.find('child').get('link'):j for j in joints if j.find('child') is not None}
    chain=[];link='link6'
    while link in by_child:
        j=by_child[link]; chain.append(j);link=j.find('parent').get('link')
    chain.reverse()
    names=[j.get('name') for j in chain]
    if not set(J).issubset(names):
        raise RuntimeError(f'Expected six UF850 joints in link6 chain, got {names}')
    limits={}; pieces=[]
    for j in chain:
        o=j.find('origin');a=j.find('axis'); lim=j.find('limit')
        T=transform(xyz(o.get('xyz') if o is not None else None),xyz(o.get('rpy') if o is not None else None))
        pieces.append((j.get('name'),j.get('type'),T,xyz(a.get('xyz') if a is not None else '1 0 0')))
        if j.get('name') in J:
            limits[j.get('name')]=(float(lim.get('lower',-math.pi)),float(lim.get('upper',math.pi))) if lim is not None else (-math.pi,math.pi)
    lo=np.array([limits[n][0] for n in J]);hi=np.array([limits[n][1] for n in J])
    return pieces,lo,hi

def fk(q,chain,base,tip):
    T=transform(np.array(base),np.zeros(3)); angles=dict(zip(J,q))
    for name,kind,origin,axis in chain:
        T=T@origin
        if kind in ('revolute','continuous'):
            u=axis/np.linalg.norm(axis);x,y,z=u;c=math.cos(angles.get(name,0));s=math.sin(angles.get(name,0))
            R=c*np.eye(3)+(1-c)*np.outer(u,u)+s*np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            A=np.eye(4);A[:3,:3]=R;T=T@A
        elif kind=='prismatic':
            A=np.eye(4);A[:3,3]=axis*angles.get(name,0);T=T@A
    return T[:3,3]+T[:3,:3]@tip,T[:3,:3]@np.array([0.,0.,1.])

def read_joints():
    import time,rclpy
    from sensor_msgs.msg import JointState
    from rclpy.node import Node
    rclpy.init();node=Node('uf850_ik_probe_readonly');value=[None]
    def callback(msg):
        d=dict(zip(msg.name,msg.position))
        if all(j in d for j in J):value[0]=np.array([d[j] for j in J])
    node.create_subscription(JointState,'/joint_states',callback,10)
    until=time.monotonic()+6
    while value[0] is None and time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.2)
    node.destroy_node();rclpy.shutdown()
    if value[0] is None:raise RuntimeError('No UF850 /joint_states received')
    return value[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--urdf',default='/workspace/simulation/generated/uf850_vgc10.urdf')
    ap.add_argument('--read-joints',action='store_true')
    ap.add_argument('--spawn',nargs=3,type=float,default=[-.05,-.47,.78])
    ap.add_argument('--box',nargs=3,type=float,default=[-.55,-.55,.82])
    ap.add_argument('--tip-offset',nargs=3,type=float,default=[0,0,.127])
    ap.add_argument('--seeds',type=int,default=20)
    args=ap.parse_args()
    chain,lo,hi=load_chain(args.urdf)
    q=read_joints() if args.read_joints else np.clip(np.zeros(6),lo,hi)
    tip=np.array(args.tip_offset)
    current,normal=fk(q,chain,args.spawn,tip)
    target=np.array(args.box)+np.array([0,0,.03+.12]) # 6-cm-tall box, 12-cm hover
    goal=np.r_[target,.20*np.array([0,0,-1.])]
    def features(q):
        p,n=fk(q,chain,args.spawn,tip)
        return np.r_[p,.20*n]
    best=(float('inf'),q)
    rng=np.random.default_rng(42)
    starts=[q,np.clip(np.zeros(6),lo,hi)]+[rng.uniform(lo,hi) for _ in range(max(0,args.seeds-2))]
    for initial in starts:
        x=initial.copy()
        for _ in range(130):
            v=features(x);err=goal-v
            score=np.linalg.norm(err)
            if score<best[0]:best=(score,x.copy())
            p,n=fk(x,chain,args.spawn,tip)
            if np.linalg.norm(p-target)<.012 and -n[2]>.975:best=(score,x.copy());break
            jac=np.column_stack([(features(np.clip(x+np.eye(6)[i]*1e-4,lo,hi))-v)/1e-4 for i in range(6)])
            dx=jac.T@np.linalg.solve(jac@jac.T+.05**2*np.eye(6),err)
            dx=np.clip(dx,-.16,.16)
            for step in (1,.5,.25):
                trial=np.clip(x+step*dx,lo,hi)
                if np.linalg.norm(goal-features(trial))<score:x=trial;break
            else:break
    p,n=fk(best[1],chain,args.spawn,tip)
    print('READ-ONLY: No commands sent to robot or vacuum')
    print('Current cup tip XYZ (world, assumed base pose):',current.round(4).tolist())
    print('Current down_dot:',round(float(-normal[2]),4))
    print('Hover target XYZ:',target.round(4).tolist())
    print('Candidate joints (rad):',best[1].round(4).tolist())
    print('FK position error (m):',round(float(np.linalg.norm(p-target)),4))
    print('Candidate down_dot:',round(float(-n[2]),4))
    print('IK result:','KINEMATIC CANDIDATE' if np.linalg.norm(p-target)<.012 and -n[2]>.975 else 'NO FEASIBLE CANDIDATE FOUND')
    print('CAUTION: No collision checking; the base pose and cup offset are assumptions.')

if __name__=='__main__':main()
