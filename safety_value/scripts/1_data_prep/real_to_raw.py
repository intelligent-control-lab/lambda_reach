#!/usr/bin/env python3
"""
Convert real-robot experiment demos into sim-format safety-analysis raw data.

Real logs (logs/real_exp/<demo>/<ts>_velocity/records.{npy,csv}) have the network obs
(obs_0..N) and a tilt-only `safety_signal`, but NO event/segment flags (and, for avoid,
no collision component in the recorded signal). This script reconstructs those so the
real data can be processed/evaluated with the same pipeline as the sim tasks.

Two task types:
  push  -> real_push   (480-dim obs; safety_signal_balance = tilt; events = ang_vel spikes;
                        contiguous segmentation, one HDF5 episode per demo)
  avoid -> real_avoid  (510-dim obs; safety_signal_collision = max(tilt, contact),
                        contact=+1 if ||ball||<=contact_dist else -1; segments = throw
                        windows [onset..landing], prep excluded, one HDF5 episode per throw)

Default mode = PREVIEW: only writes annotated full-demo segmentation plots for review.
Pass --write to also assemble logs/safety_analysis/<root>/data_raw.hdf5.

Run inside the IsaacLab/hj venv, in bash:
  source /home/ruic/IsaacLab/hj/bin/activate
  python safety_value/scripts/1_data_prep/real_to_raw.py --task avoid --demos demo_avoid_1 demo_avoid_2
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DT = 0.02
PHI_MAX = np.pi / 4.0  # 45 deg, matches sim recorders

# ----------------------------- loading -----------------------------
def find_record_subdir(demo_dir):
    for d in sorted(os.listdir(demo_dir)):
        p = os.path.join(demo_dir, d)
        if os.path.isdir(p) and (os.path.exists(os.path.join(p, "records.npy"))
                                 or os.path.exists(os.path.join(p, "records.csv"))):
            return p
    raise FileNotFoundError(f"no records.{{npy,csv}} under {demo_dir}")

def load_demo(demo_dir):
    """Return (cols: dict name->float array, obs: [T,D] float32, src)."""
    sub = find_record_subdir(demo_dir)
    cols = {}
    npy = os.path.join(sub, "records.npy")
    if os.path.exists(npy):
        a = np.load(npy, allow_pickle=True)
        names = list(a["numeric_columns"]); data = a["numeric_data"]
        for i, c in enumerate(names):
            cols[c] = np.asarray(data[:, i], dtype=np.float64)
        src = "npy"
    else:
        with open(os.path.join(sub, "records.csv")) as f:
            rd = csv.reader(f); hdr = next(rd); rows = [r for r in rd if len(r) == len(hdr)]
        for i, h in enumerate(hdr):
            try:
                cols[h] = np.array([float(r[i]) for r in rows], dtype=np.float64)
            except ValueError:
                pass  # string column (obs_key)
        src = "csv"
    obs_names = sorted([c for c in cols if c.startswith("obs_")], key=lambda s: int(s.split("_")[1]))
    obs = np.stack([cols[c] for c in obs_names], axis=1).astype(np.float32)
    return cols, obs, sub, src

# ----------------------------- signals -----------------------------
def tilt_signal(cols):
    """(tilt_deg, l_tilt) with l_tilt=(tilt-phi_max)/phi_max; >0 unsafe."""
    gx, gy, gz = cols["projected_gravity_0"], cols["projected_gravity_1"], cols["projected_gravity_2"]
    tilt = np.arctan2(np.sqrt(gx**2 + gy**2), -gz)  # rad
    return np.degrees(tilt), (tilt - PHI_MAX) / PHI_MAX

def ball_kinematics(obs):
    pos = obs[:, 492:495].astype(np.float64)   # current ball pos in base frame (newest frame)
    vel = obs[:, 507:510].astype(np.float64)   # current ball vel in base frame
    dist = np.linalg.norm(pos, axis=1)
    speed = np.linalg.norm(vel, axis=1)
    return pos, vel, dist, speed

# ----------------------------- detection -----------------------------
def detect_push_events(cols, thr=1.0, refractory_s=0.4):
    """Rising-edge onsets of |root_ang_vel| above thr, debounced by refractory."""
    av = np.sqrt(cols["root_ang_vel_0"]**2 + cols["root_ang_vel_1"]**2 + cols["root_ang_vel_2"]**2)
    above = av > thr
    onsets = []
    refr = int(round(refractory_s / DT))
    last = -10**9
    for t in range(len(av)):
        if above[t] and not above[t - 1] if t > 0 else (above[t] and t == 0):
            if t - last >= refr:
                onsets.append(t); last = t
    return np.array(onsets, dtype=int), av

def detect_throws(dist, speed, z, v_on=1.5, v_off=0.5, gap_s=0.4, min_dur_s=0.2,
                  z_ground=None, land_pad_s=0.3):
    """Return list of (onset, land) throw windows from ball kinematics.

    Flight = speed>v_on (gaps< gap_s bridged, runs< min_dur_s dropped). onset=flight start.
    landing = after flight peak, first step where ball settles on ground (z<=z_ground & speed<v_off);
    fallback = flight end + small pad. Prep (between/around throws) is excluded.
    """
    T = len(speed)
    if z_ground is None:
        z_ground = np.percentile(z, 15)  # adaptive ground level
    active = speed > v_on
    # bridge short gaps
    gap = int(round(gap_s / DT)); min_dur = int(round(min_dur_s / DT)); pad = int(round(land_pad_s / DT))
    runs = []
    t = 0
    while t < T:
        if active[t]:
            a = t
            while t < T and active[t]:
                t += 1
            b = t - 1
            runs.append([a, b])
        else:
            t += 1
    # bridge gaps
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    throws = []
    for a, b in merged:
        if (b - a + 1) < min_dur:
            continue
        # landing: from speed-peak onward, first settled-on-ground step
        peak = a + int(np.argmax(speed[a:b + 1]))
        land = b
        for t in range(peak, min(T, b + pad + 1)):
            if z[t] <= z_ground and speed[t] < v_off:
                land = t; break
        else:
            land = min(T - 1, b + pad)
        throws.append((a, land))
    return throws, z_ground

# ----------------------------- plotting -----------------------------
def shade(ax, x0, x1, color, alpha=0.15, label=None):
    ax.axvspan(x0 * DT, x1 * DT, color=color, alpha=alpha, lw=0, label=label)

def plot_push(name, cols, av, events, l_tilt, ss_rec, sv_log, a0, a1, out):
    T = len(l_tilt); t = np.arange(T) * DT
    fig, ax = plt.subplots(3, 1, figsize=(15, 7.5), sharex=True)
    # trimmed (excluded) region in gray; kept active window segmented contiguously at events
    ev_in = [e for e in events if a0 <= e < a1]
    for axi in ax:
        if a0 > 0: shade(axi, 0, a0, 'gray', 0.12)
        if a1 < T: shade(axi, a1, T, 'gray', 0.12)
    bounds = [a0] + list(ev_in) + [a1]
    for i in range(len(bounds) - 1):
        shade(ax[0], bounds[i], bounds[i + 1], ['#cfe8ff', '#ffe8cf'][i % 2], 0.5)
    ax[0].plot(t, av, lw=0.6); ax[0].axhline(1.0, color='r', ls='--', lw=1, label='ang_vel thr=1.0')
    for e in events: ax[0].axvline(e * DT, color='darkorange', lw=1.2, alpha=0.8)
    ax[0].set_ylabel("|root_ang_vel|"); ax[0].grid(alpha=.3)
    ax[0].set_title(f"PUSH {name}: gray=trimmed | shaded=segments | orange=push event | {len(ev_in)} events kept, window [{a0*DT:.0f},{a1*DT:.0f}]s")
    ax[0].legend(loc='upper right', fontsize=8)
    deg = l_tilt * 45.0 + 45.0
    ax[1].plot(t, deg, lw=0.6, color='purple'); ax[1].axhline(45, color='r', ls='--', lw=1, label='phi_max 45deg')
    ax[1].set_ylabel("tilt deg"); ax[1].grid(alpha=.3); ax[1].legend(loc='upper right', fontsize=8)
    ax[2].plot(t, l_tilt, lw=0.7, label='safety_signal (recon tilt)')
    if ss_rec is not None: ax[2].plot(t, ss_rec, lw=0.5, alpha=.5, label='safety_signal (recorded)')
    if sv_log is not None: ax[2].plot(t, sv_log, lw=0.5, alpha=.6, color='green', label='safety_value (logged)')
    ax[2].axhline(0, color='k', lw=.6)
    unsafe = l_tilt > 0
    if unsafe.any():
        ax[2].fill_between(t, -1.2, 1.4, where=unsafe, color='red', alpha=0.25, label='unsafe (l>0)')
    ax[2].set_ylabel("signal/value"); ax[2].set_xlabel("s"); ax[2].set_ylim(-1.2, 1.4)
    ax[2].grid(alpha=.3); ax[2].legend(loc='upper right', fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()

def plot_avoid(name, dist, z, speed, l_tilt, contact, l_final, sv_log, throws, z_ground, out):
    T = len(dist); t = np.arange(T) * DT
    fig, ax = plt.subplots(4, 1, figsize=(15, 9.5), sharex=True)
    # shade prep (gray) over whole, then throw windows (green) on top
    shade(ax[0], 0, T, 'gray', 0.10, 'prep (excluded)')
    for k, (a, b) in enumerate(throws):
        for axi in ax: shade(axi, a, b, 'green', 0.16)
        ax[0].axvline(a * DT, color='blue', lw=1.1, alpha=.8)
        ax[0].axvline(b * DT, color='red', lw=1.1, alpha=.8)
    ax[0].plot(t, dist, lw=0.6, label='|ball| (current)'); ax[0].axhline(0.30, color='r', ls='--', lw=1, label='contact 0.30m')
    ax[0].axhline(0.5, color='orange', ls=':', lw=1, label='danger 0.5m')
    ax[0].set_ylabel("ball dist m"); ax[0].grid(alpha=.3)
    ax[0].set_title(f"AVOID {name}: green=throw window (segment) | blue=onset red=landing | gray=prep | {len(throws)} throws")
    ax[0].legend(loc='upper right', fontsize=8)
    ax[1].plot(t, z, lw=0.6, color='green'); ax[1].axhline(z_ground, color='brown', ls='--', lw=1, label=f'z_ground={z_ground:.2f}')
    ax[1].set_ylabel("ball z (base)"); ax[1].grid(alpha=.3); ax[1].legend(loc='upper right', fontsize=8)
    ax[2].plot(t, speed, lw=0.6, color='brown'); ax[2].axhline(1.5, color='r', ls='--', lw=1, label='throw v_on=1.5')
    ax[2].set_ylabel("ball speed"); ax[2].grid(alpha=.3); ax[2].legend(loc='upper right', fontsize=8)
    ax[3].plot(t, l_tilt, lw=0.5, alpha=.6, label='tilt comp')
    ax[3].plot(t, contact, lw=0.5, alpha=.6, label='contact comp (<=0.3)')
    ax[3].plot(t, l_final, lw=0.8, color='black', label='safety_signal = max(tilt,contact)')
    if sv_log is not None: ax[3].plot(t, sv_log, lw=0.5, alpha=.6, color='magenta', label='safety_value (logged)')
    ax[3].axhline(0, color='k', lw=.6)
    unsafe = l_final > 0
    if unsafe.any(): ax[3].fill_between(t, -1.2, 1.4, where=unsafe, color='red', alpha=0.3)
    ax[3].set_ylabel("signal/value"); ax[3].set_xlabel("s"); ax[3].set_ylim(-1.2, 1.4)
    ax[3].grid(alpha=.3); ax[3].legend(loc='upper right', fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()

def plot_avoid_zoom(name, dist, z, speed, l_final, sv_log, throws, plots_dir, n_zoom):
    """Per-throw zoom plots to validate onset/landing/contact detection."""
    for k, (a, b) in enumerate(throws[:n_zoom]):
        lo = max(0, a - 150); hi = min(len(dist), b + 150)  # +/-3s context
        t = np.arange(lo, hi) * DT
        fig, ax = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
        for axi in ax: shade(axi, a, b, 'green', 0.18)
        ax[0].plot(t, dist[lo:hi], lw=1.0); ax[0].axhline(0.30, color='r', ls='--', lw=1, label='contact 0.30m')
        ax[0].axvline(a*DT, color='blue', lw=1.3, label='onset'); ax[0].axvline(b*DT, color='red', lw=1.3, label='landing')
        ax[0].set_ylabel("ball dist"); ax[0].grid(alpha=.3); ax[0].legend(loc='upper right', fontsize=8)
        ax[0].set_title(f"AVOID {name} throw {k}: window [{a*DT:.1f},{b*DT:.1f}]s")
        ax[1].plot(t, z[lo:hi], lw=1.0, color='green', label='ball z'); ax[1].plot(t, speed[lo:hi], lw=1.0, color='brown', label='speed')
        ax[1].axhline(1.5, color='r', ls=':', lw=1); ax[1].set_ylabel("z / speed"); ax[1].grid(alpha=.3); ax[1].legend(loc='upper right', fontsize=8)
        ax[2].plot(t, l_final[lo:hi], lw=1.2, color='black', label='safety_signal')
        if sv_log is not None: ax[2].plot(t, sv_log[lo:hi], lw=1.0, color='magenta', alpha=.7, label='safety_value(logged)')
        ax[2].axhline(0, color='k', lw=.6); ax[2].set_ylabel("signal/value"); ax[2].set_xlabel("s"); ax[2].set_ylim(-1.2, 1.4)
        ax[2].grid(alpha=.3); ax[2].legend(loc='upper right', fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, f"{name}_zoom_{k:02d}.png"), dpi=110); plt.close()

# ----------------------------- per-demo processing -----------------------------
def process_push(demo_dir, name, plots_dir, args):
    cols, obs, sub, src = load_demo(demo_dir)
    deg, l_tilt = tilt_signal(cols)
    ss_rec = cols.get("safety_signal"); sv_log = cols.get("safety_value")
    events, av = detect_push_events(cols, thr=args.push_thr, refractory_s=args.push_refractory)
    T = obs.shape[0]
    pad = int(round(args.push_pad / DT))
    if args.push_trim and len(events) > 0:
        a0 = max(0, int(events[0]) - pad); a1 = min(T, int(events[-1]) + pad + 1)
    else:
        a0, a1 = 0, T
    ev_local = [int(e - a0) for e in events if a0 <= e < a1]
    rec_diff = float(np.max(np.abs(l_tilt - ss_rec))) if ss_rec is not None else float("nan")
    print(f"  [{name}] T={T} obs={obs.shape[1]} src={src} | events={len(events)} | window=[{a0},{a1}) kept={a1-a0} | "
          f"unsafe(l_tilt>0,45deg) full={int((l_tilt>0).sum())} kept={int((l_tilt[a0:a1]>0).sum())} | max|recon-recorded ss|={rec_diff:.4f}")
    os.makedirs(plots_dir, exist_ok=True)
    plot_push(name, cols, av, events, l_tilt, ss_rec, sv_log, a0, a1, os.path.join(plots_dir, f"{name}_seg.png"))
    n = a1 - a0
    event_flag = np.zeros(n, dtype=np.float32)
    for e in ev_local: event_flag[e] = 1.0
    return dict(policy_obs=obs[a0:a1], safety_signal_balance=l_tilt[a0:a1].astype(np.float32),
                event_push=event_flag, terminal_state=np.zeros(n, np.float32))

def process_avoid(demo_dir, name, plots_dir, args):
    cols, obs, sub, src = load_demo(demo_dir)
    deg, l_tilt = tilt_signal(cols)
    pos, vel, dist, speed = ball_kinematics(obs)
    z = pos[:, 2]
    contact = np.where(dist <= args.contact_dist, 1.0, -1.0)
    l_final = np.maximum(l_tilt, contact)
    sv_log = cols.get("safety_value")
    throws, z_ground = detect_throws(dist, speed, z, v_on=args.throw_von, v_off=args.throw_voff,
                                     gap_s=args.throw_gap, min_dur_s=args.throw_mindur)
    T = obs.shape[0]
    print(f"  [{name}] T={T} obs={obs.shape[1]} src={src} | throws={len(throws)} | "
          f"contact(<= {args.contact_dist}m)={int((contact>0).sum())} | unsafe(l>0)={int((l_final>0).sum())} | "
          f"ball dist min={dist.min():.2f}")
    os.makedirs(plots_dir, exist_ok=True)
    plot_avoid(name, dist, z, speed, l_tilt, contact, l_final, sv_log, throws, z_ground,
               os.path.join(plots_dir, f"{name}_seg.png"))
    plot_avoid_zoom(name, dist, z, speed, l_final, sv_log, throws, plots_dir, args.n_zoom)
    # one HDF5 episode per throw window (prep excluded)
    episodes = {}
    for k, (a, b) in enumerate(throws):
        sl = slice(a, b + 1); n = b - a + 1
        episodes[f"{name}_throw_{k:02d}"] = dict(
            policy_obs=obs[sl], safety_signal_collision=l_final[sl].astype(np.float32),
            event_ball_spawn=np.zeros(n, np.float32), terminal_state=np.zeros(n, np.float32))
    return episodes

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["push", "avoid"], required=True)
    ap.add_argument("--demos", nargs="+", required=True, help="demo dir names under --real_root")
    ap.add_argument("--real_root", default="logs/real_exp")
    ap.add_argument("--out_root", default=None, help="safety_analysis root (default real_push/real_avoid)")
    ap.add_argument("--write", action="store_true", help="also write data_raw.hdf5 (default: preview plots only)")
    ap.add_argument("--push_thr", type=float, default=1.0)
    ap.add_argument("--push_refractory", type=float, default=0.4)
    ap.add_argument("--push_trim", action="store_true", default=True, help="trim to [first_event-pad, last_event+pad]")
    ap.add_argument("--no_push_trim", dest="push_trim", action="store_false")
    ap.add_argument("--push_pad", type=float, default=1.0, help="seconds of pad around push window")
    ap.add_argument("--n_zoom", type=int, default=4, help="number of per-throw zoom plots (avoid)")
    ap.add_argument("--contact_dist", type=float, default=0.30)
    ap.add_argument("--throw_von", type=float, default=1.5)
    ap.add_argument("--throw_voff", type=float, default=0.5)
    ap.add_argument("--throw_gap", type=float, default=0.4)
    ap.add_argument("--throw_mindur", type=float, default=0.2)
    args = ap.parse_args()

    root = args.out_root or ("real_push" if args.task == "push" else "real_avoid")
    out_dir = os.path.join("logs", "safety_analysis", root)
    plots_dir = os.path.join(out_dir, "seg_preview")
    print(f"== task={args.task} root={root} demos={args.demos} write={args.write}")

    all_eps = {}
    for name in args.demos:
        demo_dir = os.path.join(args.real_root, name)
        if args.task == "push":
            all_eps[name] = process_push(demo_dir, name, plots_dir, args)
        else:
            all_eps.update(process_avoid(demo_dir, name, plots_dir, args))
    print(f"  plots -> {plots_dir}")
    print(f"  total episodes that would be written: {len(all_eps)}")

    if args.write:
        import h5py
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "data_raw.hdf5")
        with h5py.File(path, "w") as f:
            g = f.create_group("data")
            for ep_name, d in all_eps.items():
                ge = g.create_group(ep_name)
                for k, v in d.items():
                    ge.create_dataset(k, data=v)
        print(f"  WROTE {path} ({len(all_eps)} episodes)")

if __name__ == "__main__":
    main()
