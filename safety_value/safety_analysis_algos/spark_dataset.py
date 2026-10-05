"""Episode-held-out SPARK datasets without quadratic future-suffix storage."""

import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeSample(tuple):
    def __new__(cls, values, future_terminal):
        obj = super().__new__(cls, values)
        obj.future_terminal = bool(future_terminal)
        return obj


def dataset_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_episode_split(path, output, *, seed=0, test_fraction=.2):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between zero and one")
    output = Path(output)
    digest = dataset_hash(path)
    if output.exists():
        split = json.loads(output.read_text())
        if split["dataset_sha256"] != digest:
            raise ValueError("Split manifest belongs to a different dataset")
        with h5py.File(path, "r") as f:
            names = set(f["episodes"])
        train, test = split["train"], split["test"]
        if (not train or not test or set(train) & set(test) or
                len(train) != len(set(train)) or len(test) != len(set(test)) or
                set(train) | set(test) != names):
            raise ValueError("Split must partition complete episodes without overlap or duplicates")
        return split
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema_version") != "spark.trajectory.v1":
            raise ValueError("Unsupported SPARK trajectory schema")
        names = sorted(f["episodes"])
        if len(names) < 2:
            raise ValueError("At least two independent episodes are required for a held-out split")
        for name in names:
            if not f["episodes"][name].attrs.get("complete", False):
                raise ValueError(f"Incomplete episode: {name}")
    order = np.random.default_rng(seed).permutation(names).tolist()
    count = min(len(order) - 1, max(1, int(round(len(order) * test_fraction))))
    split = dict(schema="lambda.episode_split.v1", dataset_sha256=digest, seed=int(seed),
                 train=sorted(order[count:]), test=sorted(order[:count]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2) + "\n")
    return split


class SparkEpisodeDataset(Dataset):
    def __init__(self, path, episode_ids, *, segment_events=True, endpoints="bootstrap"):
        if endpoints not in {"bootstrap", "legacy"}:
            raise ValueError("endpoints must be bootstrap or legacy")
        self.endpoints = endpoints
        self.segments, self.index = [], []
        with h5py.File(path, "r") as f:
            if f.attrs.get("schema_version") != "spark.trajectory.v1":
                raise ValueError("Unsupported SPARK trajectory schema")
            self.metadata = json.loads(f.attrs["metadata_json"])
            for name in episode_ids:
                group = f["episodes"][name]
                if not group.attrs.get("complete", False):
                    raise ValueError(f"Incomplete episode: {name}")
                transitions = group["transitions"]
                obs = np.asarray(transitions["policy_obs"], np.float32)
                signal = np.asarray(group["derived/safety_signal"], np.float32).reshape(-1)
                flags = np.asarray(transitions["terminated"], bool)
                events = np.flatnonzero(transitions["event_onset"][:]) if segment_events else []
                n = len(obs)
                if obs.ndim != 2 or n < 1:
                    raise ValueError(f"Expected a nonempty T-by-D observation matrix: {name}")
                if n != len(signal) or any(len(d) != n for d in transitions.values()) or any(
                        len(d) != n for d in group["derived"].values()):
                    raise ValueError(f"Inconsistent transition lengths: {name}")
                truncation = np.asarray(transitions["truncated"], bool)
                if (np.any(flags & truncation) or np.any(flags[:-1] | truncation[:-1]) or
                        not (flags[-1] or truncation[-1])):
                    raise ValueError(f"Invalid episode boundary flags: {name}")
                if any(len(d) != n + 1 for d in group["states"].values()):
                    raise ValueError(f"Expected T+1 state rows: {name}")
                if not np.isfinite(obs).all() or not np.isfinite(signal).all():
                    raise ValueError(f"Non-finite episode: {name}")
                cuts = sorted(set([0, n, *map(int, events)]))
                for start, stop in zip(cuts, cuts[1:]):
                    if stop - start < 2:
                        continue
                    x, ell = obs[start:stop], signal[start:stop]
                    future_max = np.maximum.accumulate(ell[::-1])[::-1].copy()
                    terminal = bool(flags[stop - 1]) if endpoints == "bootstrap" else True
                    index = len(self.segments)
                    self.segments.append((x, ell, future_max, terminal))
                    self.index.extend((index, i) for i in range(len(x) - 1))
        if not self.index:
            raise ValueError("Selected episodes contain no valid successor pairs")
        self.input_dim = self.segments[0][0].shape[1]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        segment, i = self.index[index]
        obs, ell, future_max, terminal = self.segments[segment]
        values = (torch.from_numpy(obs[i]), torch.from_numpy(obs[i + 1]),
                  torch.tensor([ell[i]]), torch.tensor([future_max[i]]),
                  torch.tensor([float(future_max[i] <= 0)], dtype=torch.float32),
                  torch.from_numpy(obs[i + 1:]), torch.from_numpy(ell[i + 1:]))
        return EpisodeSample(values, terminal)


def export_legacy(path, output):
    """Explicit adapter for existing preprocessing/evaluation, preserving metadata."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as source, h5py.File(output, "x") as dest:
        if source.attrs.get("schema_version") != "spark.trajectory.v1":
            raise ValueError("Unsupported SPARK trajectory schema")
        dest.attrs["spark_source_sha256"] = dataset_hash(path)
        dest.attrs["alignment"] = source.attrs["alignment"]
        data = dest.create_group("data")
        for i, name in enumerate(sorted(source["episodes"])):
            episode = source["episodes"][name]
            if not episode.attrs.get("complete", False):
                raise ValueError(f"Incomplete episode: {name}")
            group = data.create_group(f"demo_{i}")
            group.attrs["spark_episode"] = name
            for target, origin in {
                "policy_obs": "transitions/policy_obs", "terminal_state": "transitions/terminated",
                "truncated": "transitions/truncated", "event_ball_spawn": "transitions/event_onset",
                "safety_signal_spark": "derived/safety_signal",
            }.items():
                group.create_dataset(target, data=episode[origin][:])
            if "safety_value" in episode["derived"]:
                group.create_dataset("safety_value", data=episode["derived/safety_value"][:])
