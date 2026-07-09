#!/usr/bin/env python3
"""
Kinematic MuJoCo replay of a real G1 demo -> mp4, for manual segment annotation.

Uses unitree_rl_gym's g1_29dof.xml (same model/joint order as the sim2sim deploy that
produced the real data), so recorded joint_pos_0..28 map 1:1 to the model's hinge joints.
Base position is fixed (kinematic playback, no physics); base orientation from root_quat;
for avoid, a ball is placed from the base-frame ball obs (obs_492:495).

Video is real-time (output fps = 50/decim, taking every `decim`-th 50 Hz sample), so the
HTML5 video currentTime in seconds == demo time in seconds == data_index/50.

Run in bash inside the IsaacLab/hj venv:
  MUJOCO_GL=egl python safety_value/scripts/1_data_prep/replay_render.py --task avoid --demo demo_avoid_1
"""
import argparse, csv, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio.v2 as imageio

G1XML = "/home/ruic/unitree_rl_gym/resources/robots/g1_description/g1_29dof.xml"
GDIR = os.path.dirname(G1XML)
REAL_ROOT = "logs/real_exp"
OUT_DIR = "logs/real_exp_replay"
BASE_H = 0.793
DT = 0.02

SCENE = """<mujoco model="g1_replay_scene">
  <include file="g1_29dof.xml"/>
  <statistic center="0 0 0.8" extent="1.6"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4"/><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="_floor" type="plane" size="6 6 0.1" rgba="0.85 0.9 0.85 1"/>
    {ball}
  </worldbody>
</mujoco>
"""
BALL = '<body name="_ball" mocap="true" pos="2 0 1"><geom name="_ballg" type="sphere" size="0.1" rgba="0.9 0.15 0.15 1" contype="0" conaffinity="0"/></body>'

def load_demo(name):
    demo_dir = os.path.join(REAL_ROOT, name)
    sub = None
    for d in sorted(os.listdir(demo_dir)):
        p = os.path.join(demo_dir, d)
        if os.path.isdir(p) and (os.path.exists(os.path.join(p, "records.npy")) or os.path.exists(os.path.join(p, "records.csv"))):
            sub = p; break
    cols = {}
    npy = os.path.join(sub, "records.npy")
    if os.path.exists(npy):
        a = np.load(npy, allow_pickle=True)
        for i, c in enumerate(list(a["numeric_columns"])):
            cols[c] = np.asarray(a["numeric_data"][:, i], np.float64)
    else:
        with open(os.path.join(sub, "records.csv")) as f:
            rd = csv.reader(f); hdr = next(rd); rows = [r for r in rd if len(r) == len(hdr)]
        for i, h in enumerate(hdr):
            try: cols[h] = np.array([float(r[i]) for r in rows], np.float64)
            except ValueError: pass
    obs_names = sorted([c for c in cols if c.startswith("obs_")], key=lambda s: int(s.split("_")[1]))
    obs = np.stack([cols[c] for c in obs_names], axis=1) if obs_names else None
    return cols, obs

def detect_quat(cols):
    """Return [N,4] wxyz base quat, choosing the recorded column order that best
    reproduces the recorded projected_gravity = R(q)^T @ [0,0,-1]."""
    q_raw = np.stack([cols[f"root_quat_{i}"] for i in range(4)], axis=1)
    pg = np.stack([cols[f"projected_gravity_{i}"] for i in range(3)], axis=1)
    cands = {"wxyz": q_raw, "xyzw": q_raw[:, [3, 0, 1, 2]]}
    best, berr = None, 1e9
    g = np.array([0.0, 0.0, -1.0])
    for name, q in cands.items():
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        err = 0.0
        for i in range(0, len(qn), max(1, len(qn)//200)):
            res = np.zeros(3); conj = np.zeros(4)
            mujoco.mju_negQuat(conj, qn[i]); mujoco.mju_rotVecQuat(res, g, conj)
            err += np.linalg.norm(res - pg[i])
        if err < berr: berr, best = err, (name, qn)
    print(f"  quat order = {best[0]} (proj_grav fit err≈{berr:.3f})")
    return best[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["push", "avoid"], required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--decim", type=int, default=2, help="take every Nth 50Hz sample (2 -> 25fps real-time)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--azimuth", type=float, default=120)
    ap.add_argument("--elevation", type=float, default=-12)
    ap.add_argument("--distance", type=float, default=3.6)
    ap.add_argument("--sample_frames", type=int, default=0, help="also dump N sample PNG frames for verification")
    args = ap.parse_args()

    cols, obs = load_demo(args.demo)
    N = len(cols["root_quat_0"]); fps = round(50.0 / args.decim)
    quat = detect_quat(cols)
    jpos = np.stack([cols[f"joint_pos_{i}"] for i in range(29)], axis=1)

    ball_world = None
    if args.task == "avoid":
        ball_base = obs[:, 492:495]
        ball_world = np.zeros((N, 3))
        for i in range(N):
            res = np.zeros(3); mujoco.mju_rotVecQuat(res, ball_base[i], quat[i])
            ball_world[i] = np.array([0, 0, BASE_H]) + res

    scene_xml = SCENE.format(ball=BALL if args.task == "avoid" else "")
    wrap = os.path.join(GDIR, f"_replay_scene_{args.task}.xml")
    with open(wrap, "w") as f: f.write(scene_xml)
    m = mujoco.MjModel.from_xml_path(wrap)
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, height=args.height, width=args.width)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.8]; cam.distance = args.distance; cam.azimuth = args.azimuth; cam.elevation = args.elevation
    mocap_id = m.body("_ball").mocapid[0] if args.task == "avoid" else -1

    os.makedirs(OUT_DIR, exist_ok=True)
    out_mp4 = os.path.join(OUT_DIR, f"{args.demo}.mp4")
    idxs = list(range(0, N, args.decim))
    print(f"  rendering {len(idxs)} frames @ {fps}fps -> {out_mp4} ({N} steps, {N*DT:.1f}s)")
    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=6,
                                output_params=["-pix_fmt", "yuv420p"])
    sample_every = max(1, len(idxs)//args.sample_frames) if args.sample_frames else 0
    try:
        import cv2; have_cv2 = True
    except Exception:
        have_cv2 = False
    for k, i in enumerate(idxs):
        d.qpos[0:3] = [0, 0, BASE_H]
        d.qpos[3:7] = quat[i]
        d.qpos[7:36] = jpos[i]
        if mocap_id >= 0:
            d.mocap_pos[mocap_id] = ball_world[i]
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera=cam)
        px = ren.render()
        if have_cv2:
            cv2.putText(px, f"t={i*DT:6.2f}s  step={i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        writer.append_data(px)
        if sample_every and k % sample_every == 0:
            imageio.imwrite(os.path.join(OUT_DIR, f"{args.demo}_frame_{i:06d}.png"), px)
    writer.close(); ren.close()
    # write duration sidecar (seconds) so the website knows the timeline length
    with open(os.path.join(OUT_DIR, f"{args.demo}.meta"), "w") as f:
        f.write(f"{N}\n{N*DT}\n{fps}\n")
    print(f"  DONE {out_mp4}  duration={N*DT:.1f}s")

if __name__ == "__main__":
    main()
