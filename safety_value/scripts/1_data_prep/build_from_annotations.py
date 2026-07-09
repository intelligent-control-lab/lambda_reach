#!/usr/bin/env python3
"""
Build real_push / real_avoid raw data + per-segment review artifacts from manual annotations.

Reads logs/real_exp_replay/annotations.json ({demo: {task, segments:[[s,e]s,...]}}). For each
annotated segment it slices the recorded obs (index = round(t*50)), computes the ground-truth
safety signal (push: tilt@45deg; avoid: max(tilt@45deg, graded (0.5-d)/0.5 with a +1
contact plateau @ d<=0.30m)), and:
  - saves a per-segment plot (safety_signal + logged value [+ ball dist for avoid]),
  - cuts a per-segment video clip from the replay mp4,
  - (with --write) writes logs/safety_analysis/<root>/data_raw.hdf5 (one episode per segment).

Review dir: logs/safety_analysis/<root>/seg_review/.  Run in IsaacLab/hj venv (bash).
"""
import argparse, csv, json, os, subprocess
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DT = 0.02; PHI = np.pi / 4.0
REAL = "logs/real_exp"; REPLAY = "logs/real_exp_replay"; ANN = os.path.join(REPLAY, "annotations.json")
CONTACT_DIST = 0.30   # ball<=this (pelvis-frame) => contact proxy (sim uses a contact sensor)
DANGER_DIST = 0.5     # sim danger_distance: graded signal (danger-d)/danger
LATCH_S = 0.25        # sim latch_duration: hold +1 this long after a contact

def load_demo(name):
    demo_dir = os.path.join(REAL, name); sub = None
    for d in sorted(os.listdir(demo_dir)):
        p = os.path.join(demo_dir, d)
        if os.path.isdir(p) and (os.path.exists(os.path.join(p, "records.npy")) or os.path.exists(os.path.join(p, "records.csv"))):
            sub = p; break
    cols = {}
    npy = os.path.join(sub, "records.npy")
    if os.path.exists(npy):
        a = np.load(npy, allow_pickle=True)
        for i, c in enumerate(list(a["numeric_columns"])): cols[c] = np.asarray(a["numeric_data"][:, i], np.float64)
    else:
        with open(os.path.join(sub, "records.csv")) as f:
            rd = csv.reader(f); hdr = next(rd); rows = [r for r in rd if len(r) == len(hdr)]
        for i, h in enumerate(hdr):
            try: cols[h] = np.array([float(r[i]) for r in rows], np.float64)
            except ValueError: pass
    on = sorted([c for c in cols if c.startswith("obs_")], key=lambda s: int(s.split("_")[1]))
    return cols, np.stack([cols[c] for c in on], axis=1).astype(np.float32)

def tilt_l(cols):
    gx, gy, gz = cols["projected_gravity_0"], cols["projected_gravity_1"], cols["projected_gravity_2"]
    tilt = np.arctan2(np.sqrt(gx**2 + gy**2), -gz)
    return (tilt - PHI) / PHI

def ffmpeg_exe():
    try:
        from imageio_ffmpeg import get_ffmpeg_exe; return get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

def plot_seg(name, t0, l, sv, dist, out, task, ev=None):
    n = len(l); t = t0 + np.arange(n) * DT
    fig, ax = plt.subplots(1, 1, figsize=(9, 3.4))
    ax.plot(t, l, lw=1.4, color="black", label="safety_signal (GT)")
    if sv is not None: ax.plot(t, sv, lw=1.0, color="magenta", alpha=.8, label="safety_value (logged)")
    if dist is not None:
        ax2 = ax.twinx(); ax2.plot(t, dist, lw=1.0, color="teal", alpha=.6, label="ball dist")
        ax2.axhline(DANGER_DIST, color="teal", ls=":", lw=1); ax2.set_ylabel("ball dist m (0.5=unsafe bound)", color="teal")
    ax.axhline(0, color="r", ls="--", lw=1)
    if ev is not None: ax.axvline(ev, color="orange", lw=2, label="event")
    unsafe = l > 0
    if unsafe.any(): ax.fill_between(t, -1.2, 1.4, where=unsafe, color="red", alpha=.25)
    ax.set_ylim(-1.2, 1.4); ax.set_xlabel("s"); ax.set_ylabel("signal/value")
    ax.set_title("%s  (%.1fs, unsafe steps=%d/%d)" % (name, n * DT, int(unsafe.sum()), n)); ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()

def downsample(arr, m=400):
    if len(arr) <= m: return [round(float(x), 4) for x in arr]
    idx = np.linspace(0, len(arr) - 1, m).astype(int)
    return [round(float(arr[i]), 4) for i in idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["push", "avoid"], required=True)
    ap.add_argument("--write", action="store_true", help="write data_raw.hdf5")
    ap.add_argument("--no_clips", action="store_true")
    args = ap.parse_args()
    ann = json.load(open(ANN))
    root = "real_push" if args.task == "push" else "real_avoid"
    sig_key = "safety_signal_balance" if args.task == "push" else "safety_signal_collision"
    evt_key = "event_push" if args.task == "push" else "event_ball_spawn"
    out_dir = os.path.join("logs", "safety_analysis", root); rev = os.path.join(out_dir, "seg_review")
    os.makedirs(rev, exist_ok=True)
    ffe = ffmpeg_exe()
    episodes = {}; n_unsafe_segs = 0; total_steps = 0; n_events = 0; sig_entries = []
    for demo, d in ann.items():
        if d.get("task") != args.task: continue
        segs = d.get("segments", [])
        if not segs: continue
        evs = d.get("events", [])
        cols, obs = load_demo(demo); l_tilt = tilt_l(cols); sv = cols.get("safety_value")
        dist = None
        if args.task == "avoid":
            dist = np.linalg.norm(obs[:, 492:495], axis=1)
            # Graded proximity signal (danger-d)/danger in [-1, 1]: >0 (unsafe) once the ball is
            # within danger_distance (0.5m), ramping up as it nears. PLUS a hard +1 "contact"
            # plateau when within CONTACT_DIST (0.30m), matching the sim value's +1-on-contact
            # target magnitude. The unsafe *set* (sign) is still {d<0.5}, so the sign-based metrics
            # (recall/FPR/detection) are unchanged vs the graded-only signal; only value MSE shifts.
            l_coll = np.clip((DANGER_DIST - dist) / DANGER_DIST, -1.0, 1.0)
            l_coll[dist <= CONTACT_DIST] = 1.0
            l_full = np.maximum(l_tilt, l_coll)
        else:
            l_full = l_tilt
        for k, (s, e) in enumerate(segs):
            i0, i1 = int(round(s * 50)), int(round(e * 50)); i1 = min(i1, len(l_full)); n = i1 - i0
            if n < 3: print("  skip short seg", demo, k); continue
            name = "%s_seg%02d" % (demo, k); sl = slice(i0, i1)
            # event flag: marked disturbance onset within [s,e] (default = segment start)
            ev = evs[k] if (k < len(evs) and evs[k] is not None) else None
            ev_idx = 0 if ev is None else int(np.clip(round((ev - s) * 50), 0, n - 1))
            if ev is not None: n_events += 1
            ev_arr = np.zeros(n, np.float32); ev_arr[ev_idx] = 1.0
            ep = {"policy_obs": obs[sl], sig_key: l_full[sl].astype(np.float32),
                  evt_key: ev_arr, "terminal_state": np.zeros(n, np.float32)}
            episodes[name] = ep; total_steps += n
            if (l_full[sl] > 0).any(): n_unsafe_segs += 1
            plot_seg(name, s, l_full[sl], sv[sl] if sv is not None else None,
                     dist[sl] if dist is not None else None, os.path.join(rev, name + ".png"), args.task,
                     ev=ev)
            sig_entries.append({
                "demo": demo, "k": k, "name": name, "task": args.task,
                "start": round(s, 3), "end": round(e, 3), "event": (round(ev, 3) if ev is not None else None),
                "t": [round(s + j * DT, 3) for j in np.linspace(0, n - 1, min(n, 400)).astype(int)],
                "l": downsample(l_full[sl]), "v": (downsample(sv[sl]) if sv is not None else None),
                "d": (downsample(dist[sl]) if dist is not None else None),
            })
            if not args.no_clips:
                src = os.path.join(REPLAY, demo + ".mp4")
                if os.path.exists(src):
                    subprocess.run([ffe, "-y", "-loglevel", "error", "-ss", str(s), "-i", src,
                                    "-t", str(e - s), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                    os.path.join(rev, name + ".mp4")], check=False)
    json.dump({"segments": sig_entries}, open(os.path.join(REPLAY, "seg_signals_%s.json" % args.task), "w"))
    print("== %s: %d segments (%d with unsafe steps, %d events marked), %d total steps (%.1fs)" %
          (root, len(episodes), n_unsafe_segs, n_events, total_steps, total_steps * DT))
    print("   review -> %s ; seg_signals -> %s" % (rev, os.path.join(REPLAY, "seg_signals_%s.json" % args.task)))
    if args.write:
        import h5py
        path = os.path.join(out_dir, "data_raw.hdf5")
        with h5py.File(path, "w") as f:
            g = f.create_group("data")
            for nm, ep in episodes.items():
                ge = g.create_group(nm)
                for kk, vv in ep.items(): ge.create_dataset(kk, data=vv)
        print("   WROTE %s (%d episodes)" % (path, len(episodes)))

if __name__ == "__main__":
    main()
