"""Research boundary contracts: temporal alignment, cutoffs, splits and monitoring."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
import torch

from safety_value.monitoring.spark_monitor import ReachabilityObserver
from safety_value.safety_analysis_algos.spark_dataset import (
    EpisodeSample, SparkEpisodeDataset, create_episode_split, export_legacy,
)
from safety_value.safety_analysis_algos.dataset import hj_collate


def test_bundle_rejects_changed_weights_wrong_policy_and_unrelated_evaluation(tmp_path):
    from safety_value.monitoring.spark_bundle import load_bundle, sha256
    files = {name: tmp_path / name for name in
             ("policy", "config", "value_checkpoint", "value_config", "test_results")}
    for name in ("policy", "config", "value_checkpoint"):
        files[name].write_bytes(name.encode())
    config = dict(policy_sha256=sha256(files["policy"]), config_sha256=sha256(files["config"]),
                  source_manifest_sha256="dataset")
    evaluation = dict(complete=True, split="test", dataset_sha256="dataset",
                      models={"value": {"checkpoint_sha256": sha256(files["value_checkpoint"])}})
    bundle_path = tmp_path / "bundle.json"

    def write_bundle():
        files["value_config"].write_text(json.dumps(config))
        files["test_results"].write_text(json.dumps(evaluation))
        bundle = {"schema": "lambda.spark.monitor_bundle.v1"}
        bundle.update({name: dict(path=path.name, sha256=sha256(path)) for name, path in files.items()})
        bundle_path.write_text(json.dumps(bundle))

    write_bundle()
    assert load_bundle(bundle_path)["policy"] == files["policy"]
    files["policy"].write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(bundle_path)
    write_bundle()
    with pytest.raises(ValueError, match="different policy"):
        load_bundle(bundle_path)
    config["policy_sha256"] = sha256(files["policy"])
    evaluation["models"]["value"]["checkpoint_sha256"] = "unrelated"
    write_bundle()
    with pytest.raises(ValueError, match="does not include"):
        load_bundle(bundle_path)


def make_data(path, episodes=4):
    from spark_pipeline.trajectory import EpisodeWriter
    with EpisodeWriter(path, {"policy_sha256": "fixture"}) as writer:
        for episode in range(episodes):
            writer.begin({"time": 0.}, {"seed": episode, "episode_id": episode})
            for step in range(4):
                info = dict(policy_obs=np.array([episode, step], np.float32),
                            action_raw=np.zeros(1), target_actuated_pos=np.zeros(1))
                writer.append(info, {"time": (step + 1) * .02},
                              dict(terminated=False, truncated=step == 3, event_onset=step == 2),
                              derived={"safety_signal": np.float32([-.8, -.5, .5, -.1][step])})
            writer.end()


def test_episode_split_is_stable_disjoint_and_content_bound(tmp_path):
    path, split_path = tmp_path / "data.h5", tmp_path / "split.json"
    make_data(path)
    split = create_episode_split(path, split_path, seed=7)
    assert not set(split["train"]) & set(split["test"])
    assert len(split["train"]) + len(split["test"]) == 4
    assert create_episode_split(path, split_path, seed=99) == split
    with h5py.File(path, "a") as f:
        f.attrs["changed"] = True
    with pytest.raises(ValueError, match="different dataset"):
        create_episode_split(path, split_path)


def test_split_rejects_overlapping_episodes(tmp_path):
    path, split_path = tmp_path / "data.h5", tmp_path / "split.json"
    make_data(path)
    split = create_episode_split(path, split_path)
    split["test"].append(split["train"][0])
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="without overlap"):
        create_episode_split(path, split_path)


def test_dataset_segments_and_future_values_match_original_processor(tmp_path):
    path = tmp_path / "data.h5"
    make_data(path, 2)
    dataset = SparkEpisodeDataset(path, ["episode_000000"], endpoints="legacy")
    processor_path = Path(__file__).resolve().parents[1] / "safety_value/scripts/1_data_prep/data_processing.py"
    spec = importlib.util.spec_from_file_location("legacy_processor", processor_path)
    processor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(processor)
    expected = processor.process_segment(np.array([-.8, -.5], np.float32),
                                         np.zeros(2), np.array([[0, 0], [0, 1]], np.float32))
    sample = dataset[0]
    assert len(dataset) == 2
    for index in range(5):
        np.testing.assert_allclose(sample[index].numpy().reshape(-1), np.asarray(expected[index][0]).reshape(-1))
    np.testing.assert_array_equal(sample[5], expected[5][0])
    np.testing.assert_array_equal(sample[6], expected[6][0])


def test_legacy_export_keeps_post_action_signal_and_truncation(tmp_path):
    path, output = tmp_path / "data.h5", tmp_path / "legacy.h5"
    make_data(path, 2)
    export_legacy(path, output)
    with h5py.File(output) as f:
        group = f["data/demo_0"]
        np.testing.assert_array_equal(group["policy_obs"][:, 1], [0, 1, 2, 3])
        np.testing.assert_allclose(group["safety_signal_spark"], [-.8, -.5, .5, -.1])
        assert not group["terminal_state"][:].any()
        assert group["truncated"][-1]


def test_lambda_bootstraps_at_cutoff_and_preserves_legacy_terminal_rule():
    from safety_value.safety_analysis_algos.lambda_reachability.trainer import LambdaReachabilityTrainer
    values = (torch.zeros(2), torch.ones(2), torch.tensor([-.8]), torch.tensor([-.5]),
              torch.tensor([1.]), torch.ones((1, 2)), torch.tensor([-.5]))
    sample = EpisodeSample(values, False)
    trainer = LambdaReachabilityTrainer(2, [sample], hidden_dims=(4,), batch_size=1,
                                        max_horizon=1, lambda_param=0., delta=1.)
    for model in (trainer.target1, trainer.target2):
        for p in model.parameters():
            p.data.zero_()
        list(model.modules())[-1].bias.data.fill_(2.)
    assert trainer._compute_targets(hj_collate([sample]))[3].item() == 2.
    assert trainer._compute_targets(hj_collate([EpisodeSample(values, True)]))[3].item() == -.5
    assert trainer._compute_targets(hj_collate([values]))[3].item() == -.5


def state(time=0., contact=0., height=1.):
    return dict(time=time, base_tilt=0., base_height=.8, torso_contact_force=0.,
                root_pose_wxyz=np.array([0, 0, .8, 1, 0, 0, 0]),
                ball_position_w=np.array([1, 0, height]), ball_contact_force=contact)


def test_collision_latch_resets_and_ground_contact_is_filtered():
    monitor = ReachabilityObserver()
    assert monitor.signal(state(contact=2., height=.1)) < 0
    assert monitor.signal(state(time=1., contact=2.)) == 1
    assert monitor.signal(state(time=1.2)) == 1
    assert monitor.signal(state(time=1.26)) < 0
    monitor.reset()
    assert monitor.signal(state(time=1.1)) < 0


def test_observer_uses_applied_policy_input_and_does_not_mutate_action():
    class Value:
        def __call__(self, obs):
            np.testing.assert_array_equal(obs, [1., 2.])
            return np.array([.25])
    monitor = ReachabilityObserver(Value())
    action = {"policy_obs": np.array([1., 2.]), "target_actuated_pos": np.array([.4])}
    result = monitor(None, action, SimpleNamespace(agent_feedback=state()))
    assert result["safety_value"] == .25
    np.testing.assert_array_equal(action["target_actuated_pos"], [.4])


def test_viewer_observer_displays_predictions_and_resets():
    from safety_value.monitoring.spark_monitor import ReachabilityViewerObserver

    class Agent:
        def set_viewer_simulation_info(self, values):
            self.display = values

    agent = Agent()
    observer = ReachabilityViewerObserver(agent, lambda observation: np.array([.25]))
    result = observer(None, {'policy_obs': np.zeros(2)}, SimpleNamespace(agent_feedback=state()))
    assert result['safety_value'] == .25
    assert agent.display['Reachability V(x)'] == '+0.250'
    assert agent.display['Predicted state'] == 'UNSAFE'
    observer.reset()
    assert agent.display['Reachability V(x)'] == 'waiting'
