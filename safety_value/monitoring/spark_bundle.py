"""Portable, hash-checked policy/value pairs for SPARK deployment."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_bundle(path):
    path = Path(path).resolve()
    bundle = json.loads(path.read_text())
    if bundle.get("schema") != "lambda.spark.monitor_bundle.v1":
        raise ValueError("Unsupported SPARK monitor bundle")
    files = {}
    for name in ("policy", "config", "value_checkpoint", "value_config", "test_results"):
        entry = bundle[name]
        files[name] = path.parent / entry["path"]
        if sha256(files[name]) != entry["sha256"]:
            raise ValueError(f"Bundle hash mismatch: {name}")
    value_config = json.loads(files["value_config"].read_text())
    if value_config.get("policy_sha256") != bundle["policy"]["sha256"]:
        raise ValueError("Bundle value was trained for a different policy")
    if value_config.get("config_sha256") != bundle["config"]["sha256"]:
        raise ValueError("Bundle value was trained with a different deploy configuration")
    evaluation = json.loads(files["test_results"].read_text())
    if not evaluation.get("complete"):
        raise ValueError("Bundle evaluation is incomplete")
    if evaluation.get("split") != "test" or evaluation.get("dataset_sha256") != value_config.get("source_manifest_sha256"):
        raise ValueError("Bundle evaluation does not use the held-out training dataset split")
    if not any(result.get("checkpoint_sha256") == bundle["value_checkpoint"]["sha256"]
               for result in evaluation.get("models", {}).values()):
        raise ValueError("Bundle evaluation does not include this value checkpoint")
    return files
