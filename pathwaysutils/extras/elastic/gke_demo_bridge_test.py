# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for gke_demo_bridge simple_model and MaxText workload support."""

import os
import queue
import re
import subprocess
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from pathwaysutils.extras.elastic import gke_demo_bridge as bridge
import yaml

_MAXTEXT_COMMAND = (
    "python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml"
    " model_name=llama3.1-8b elastic_enabled=True"
    " elastic_min_slice_count={min_slices} elastic_backup_kind=snapshot"
    " profiler_steps=3 skip_first_n_steps_for_profiler=8 steps=3000"
)
_SIMPLE_MODEL_COMMAND = (
    "python3 -m pathwaysutils.extras.elastic.elastic_workload"
    " --total_steps=750 --elasticity_mode=step_snapshot_pause_resume"
)
_MAXTEXT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(bridge.__file__), "elastic_demo_maxtext.template.yaml"
)


def _jobset(head_command: str, replicas: int = 2) -> dict[str, Any]:
  """Returns a minimal Pathways JobSet object with the given head command."""
  return {
      "apiVersion": "jobset.x-k8s.io/v1alpha2",
      "kind": "JobSet",
      "metadata": {"name": "maxtext-demo", "namespace": "default"},
      "spec": {
          "replicatedJobs": [
              {
                  "name": "pathways-head",
                  "replicas": 1,
                  "template": {"spec": {"template": {"spec": {
                      "containers": [{
                          "name": "jax-tpu",
                          "command": ["bash", "-c", head_command],
                      }],
                  }}}},
              },
              {
                  "name": "pathways-worker",
                  "replicas": replicas,
                  "template": {"spec": {"template": {"spec": {
                      "nodeSelector": {
                          "cloud.google.com/gke-tpu-accelerator": (
                              "tpu-v5-lite-podslice"
                          ),
                          "cloud.google.com/gke-tpu-topology": "4x8",
                      },
                      "containers": [{"name": "pathways-worker"}],
                  }}}},
              },
          ]
      },
  }


def _maxtext_step(step: int, seconds: float = 4.05, live: int = 2,
                  loss: float = 6.4) -> str:
  return (
      "I1027 02:04:47.035396 133108186096448 metric_logger.py:265] completed"
      f" step: {step}, seconds: {seconds:.3f}, live slice count: {live},"
      " TFLOP/s/device: 104.0, Tokens/s/device: 2020.0, total_weights: 31051,"
      f" loss: {loss:.3f}, lm_loss: {loss:.3f}, perplexity: 600.0"
  )


class WorkloadDetectionTest(parameterized.TestCase):

  def test_detect_workload_type(self):
    self.assertEqual(
        bridge.detect_workload_type(_MAXTEXT_COMMAND.format(min_slices=-1)),
        bridge.WORKLOAD_MAXTEXT,
    )
    self.assertEqual(
        bridge.detect_workload_type("python3 -m MaxText.train MaxText/configs/"
                                    "base.yml steps=10"),
        bridge.WORKLOAD_MAXTEXT,
    )
    self.assertEqual(
        bridge.detect_workload_type(_SIMPLE_MODEL_COMMAND),
        bridge.WORKLOAD_SIMPLE_MODEL,
    )
    self.assertIsNone(bridge.detect_workload_type("python3 my_trainer.py"))

  @parameterized.named_parameters(
      ("default_waits_for_all", "-1", 2, "step_snapshot_pause_resume"),
      ("equal_to_slices", "2", 2, "step_snapshot_pause_resume"),
      ("below_slices", "1", 2, "step_snapshot_replica_resize"),
  )
  def test_parse_maxtext_command(self, min_slices, num_slices, expected_mode):
    inferred = bridge.parse_maxtext_command(
        _MAXTEXT_COMMAND.format(min_slices=min_slices), num_slices
    )
    self.assertEqual(inferred["workload_type"], bridge.WORKLOAD_MAXTEXT)
    self.assertEqual(inferred["total_steps"], 3000)
    self.assertEqual(inferred["elasticity_mode"], expected_mode)

  def test_parse_maxtext_command_checkpoint_backup(self):
    inferred = bridge.parse_maxtext_command(
        "steps=50 elastic_backup_kind=checkpoint elastic_min_slice_count=1", 4
    )
    self.assertEqual(inferred["elasticity_mode"], "checkpoint_replica_resize")

  def test_infer_config_from_maxtext_yaml(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "elastic_demo_maxtext.yaml")
      with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_jobset(_MAXTEXT_COMMAND.format(min_slices=-1)), f)
      inferred = bridge._infer_config_from_yaml(path)
    self.assertEqual(inferred["workload_type"], bridge.WORKLOAD_MAXTEXT)
    self.assertEqual(inferred["jobset_name"], "maxtext-demo")
    self.assertEqual(inferred["num_slices"], 2)
    self.assertEqual(inferred["tpu_type"], "tpuv5e")
    self.assertEqual(inferred["topology"], "4x8")
    self.assertEqual(inferred["total_steps"], 3000)
    self.assertEqual(inferred["elasticity_mode"], "step_snapshot_pause_resume")

  def test_infer_config_from_simple_model_jobset(self):
    inferred = bridge._infer_config_from_jobset_doc(
        _jobset(_SIMPLE_MODEL_COMMAND, replicas=3)
    )
    self.assertEqual(inferred["workload_type"], bridge.WORKLOAD_SIMPLE_MODEL)
    self.assertEqual(inferred["num_slices"], 3)
    self.assertEqual(inferred["total_steps"], 750)
    self.assertEqual(inferred["elasticity_mode"], "step_snapshot_pause_resume")

  def test_merge_config_rejects_unknown_workload(self):
    cfg = bridge._merge_config_dict(
        {"tpu_type": "tpuv5e", "topology": "4x8"}, {"workload_type": "bogus"}
    )
    self.assertEqual(cfg["workload_type"], bridge.WORKLOAD_SIMPLE_MODEL)

  def test_split_kubectl_timestamp(self):
    ts, line = bridge._split_kubectl_timestamp(
        "2026-10-27T02:46:07.689248268Z I1027 completed step: 543"
    )
    self.assertAlmostEqual(ts, 1793069167.689248, places=5)
    self.assertEqual(line, "I1027 completed step: 543")
    self.assertEqual(
        bridge._split_kubectl_timestamp("no timestamp"), (None, "no timestamp")
    )


class LogParsingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    with mock.patch.object(
        bridge, "_current_kubeconfig_context", return_value="test-context"
    ):
      self.state = bridge.BridgeState()
    self.tailer = bridge.LogTailer(self.state)
    self.events: queue.Queue[dict[str, Any]] = queue.Queue()
    self.state.register_subscriber(self.events)

  def _feed(self, line: str, now: float) -> None:
    self.tailer._parse_line(
        line,
        "head-pod",
        "test-context",
        "default",
        bridge._SIMPLE_MODEL_STEP_RE,
        bridge._SIMPLE_MODEL_DISRUPT_RE,
        bridge._SIMPLE_MODEL_STAGE1_RE,
        bridge._SIMPLE_MODEL_STAGE2_RE,
        now=now,
    )

  def _drain_event_names(self) -> list[str]:
    names = []
    while not self.events.empty():
      evt = self.events.get_nowait()
      names.append(evt.get("event") or evt["type"])
    return names

  def test_maxtext_pause_resume_recovery(self):
    self.state.update_config(
        {"workload_type": bridge.WORKLOAD_MAXTEXT, "num_slices": 2}
    )
    t = 1000.0
    self._feed(_maxtext_step(0, seconds=116.6, loss=12.3), t)
    self.assertEqual(self.state.jobset_status, "RUNNING")
    self.assertTrue(self.state.workload_ready)
    for step in range(1, 21):
      self._feed(_maxtext_step(step), t + 4 * step)
    self.assertEqual(self.state.current_step, 20)
    self.assertAlmostEqual(self.state.forward_progress_time, 80.0)
    goodput_before = self.state.current_goodput
    self.assertGreater(goodput_before, 90.0)
    self.assertIn("training_started", self._drain_event_names())

    self._feed("E1027 02:15:16.265599 1 manager.py:411] Elastic event detected",
               1090.0)
    self.assertEqual(self.state.jobset_status, "DISRUPTED")
    self.assertEqual(self.state.inactive_slices, [1])
    self._feed("I1027 1 manager.py:374] Elastic attempt 2", 1091.0)
    self._feed(
        "I1027 1 elastic.py:231] 1 slices active. Wanting at least 2. Active"
        " indices: frozenset({1})",
        1100.0,
    )
    self.assertEqual(self.state.jobset_status, "DISRUPTED")
    self.assertIn("waiting for capacity", self.state.current_event)

    self._feed(
        "I1027 1 elastic.py:223] Sufficient slices active: 2 >= 2. Active"
        " indices: frozenset({0, 1})",
        1200.0,
    )
    self.assertEqual(self.state.jobset_status, "RECOVERING")
    self._feed(
        "I1027 1 snapshotting.py:396] Restoring the in-memory snapshot at step"
        " 15...",
        1204.0,
    )
    self.assertEqual(self.state.current_step, 15)

    # Steps 16-20 are recomputed after the rewind and earn no goodput credit.
    for i, step in enumerate(range(16, 23)):
      self._feed(_maxtext_step(step, seconds=90.9 if step == 16 else 4.05),
                 1300.0 + 4 * i)
    self.assertEqual(self.state.jobset_status, "RUNNING")
    self.assertEqual(self.state.active_slices, [0, 1])
    self.assertAlmostEqual(self.state.forward_progress_time, 88.0)
    self.assertLess(self.state.current_goodput, 30.0)
    self.assertEqual(self.state.max_step_seen, 22)
    self.assertEqual(
        [n for n in self._drain_event_names() if n != "step"],
        [
            "disrupted",
            "recovery_started",
            "stage1",
            "stage3",
            "recovery_complete",
        ],
    )

  def test_maxtext_replica_resize_degrades_then_scales_up(self):
    self.state.update_config(
        {"workload_type": bridge.WORKLOAD_MAXTEXT, "num_slices": 2}
    )
    self._feed(_maxtext_step(0), 1000.0)
    self._feed("E1027 1 manager.py:411] Elastic event detected", 1005.0)
    self._feed(_maxtext_step(1, live=1), 1030.0)
    self.assertEqual(self.state.jobset_status, "DEGRADED")
    self.assertEqual(len(self.state.active_slices), 1)
    self._feed(
        "I1027 1 snapshotting.py:327] New slices are available. Snapshotting"
        " step 5 before scaling up.",
        1050.0,
    )
    self.assertEqual(self.state.jobset_status, "SCALING_UP")
    self._feed(_maxtext_step(2, live=1), 1052.0)
    self.assertEqual(self.state.jobset_status, "SCALING_UP")
    self._feed(_maxtext_step(3, live=2), 1100.0)
    self.assertEqual(self.state.jobset_status, "RUNNING")
    names = self._drain_event_names()
    self.assertEqual(names.count("recovery_complete"), 2)
    self.assertIn("scale_up_detected", names)

  def test_maxtext_replica_resize_with_manager_scale_up_lines(self):
    # Log lines as emitted by a live replica-resize run (MaxText 20261026).
    self.state.update_config(
        {"workload_type": bridge.WORKLOAD_MAXTEXT, "num_slices": 2}
    )
    self._feed(_maxtext_step(20), 1000.0)
    self._feed("E1027 1 manager.py:411] Elastic event detected", 1020.0)
    self._feed(
        "I1027 1 elastic.py:223] Sufficient slices active: 1 >= 1. Active"
        " indices: frozenset({0})",
        1021.0,
    )
    self.assertEqual(self.state.jobset_status, "RECOVERING")
    self._feed(
        "I1027 1 checkpointing.py:729] Restoring the checkpoint at step 20"
        " rather than the in-memory snapshot at step 15, which is older.",
        1025.0,
    )
    self._feed(
        "I1027 1 checkpointing.py:736] restoring from this run's directory"
        " step 20",
        1025.1,
    )
    self.assertEqual(self.state.current_step, 20)
    self._feed(_maxtext_step(21, seconds=57.2, live=1), 1160.0)
    self.assertEqual(self.state.jobset_status, "DEGRADED")
    self._drain_event_names()

    self._feed(
        "I1027 1 manager.py:263] Newly available but inactive slices"
        " frozenset({1})",
        1600.0,
    )
    self.assertEqual(self.state.jobset_status, "SCALING_UP")
    self._feed(
        "I1027 1 elastic_utils.py:505] Started a checkpoint and a new slice is"
        " available. Waiting for current checkpoint to finish before"
        " interrupting.",
        1608.0,
    )
    self._feed("I1027 1 manager.py:396] Scale up requested.", 1623.0)
    self.assertEqual(self.state.jobset_status, "SCALING_UP")
    self._feed(
        "I1027 1 elastic.py:223] Sufficient slices active: 2 >= 1. Active"
        " indices: frozenset({0, 1})",
        1624.0,
    )
    self.assertEqual(self.state.jobset_status, "RECOVERING")
    self._feed(_maxtext_step(41, seconds=89.1, live=2), 1756.0)
    self.assertEqual(self.state.jobset_status, "RUNNING")
    self.assertEqual(
        [n for n in self._drain_event_names() if n != "step"],
        ["scale_up_detected", "recovery_started", "stage1",
         "recovery_complete"],
    )

  def test_maxtext_ignores_empty_newly_available_set(self):
    self.state.update_config({"workload_type": bridge.WORKLOAD_MAXTEXT})
    self._feed(_maxtext_step(5), 1000.0)
    self._feed(
        "I1027 1 manager.py:263] Newly available but inactive slices"
        " frozenset()",
        1001.0,
    )
    self.assertEqual(self.state.jobset_status, "RUNNING")

  def test_simple_model_ignores_maxtext_scale_up_lines(self):
    self._feed("W1026 >>> [CONTROL] TRAINING STARTED signal received! <<<",
               1000.0)
    self._feed("I1026 manager.py:396] Scale up requested.", 1001.0)
    self.assertEqual(self.state.jobset_status, "RUNNING")

  def test_maxtext_exit_code_marks_completion(self):
    self.state.update_config({"workload_type": bridge.WORKLOAD_MAXTEXT})
    self._feed(_maxtext_step(0), 1000.0)
    self._feed("EXIT_CODE=0", 1010.0)
    self.assertEqual(self.state.jobset_status, "COMPLETED")
    self.assertFalse(self.state.training_running)

  def test_maxtext_step_switches_workload_type(self):
    self.assertEqual(
        self.state.get_config()["workload_type"], bridge.WORKLOAD_SIMPLE_MODEL
    )
    self._feed(_maxtext_step(7), 1000.0)
    self.assertEqual(
        self.state.get_config()["workload_type"], bridge.WORKLOAD_MAXTEXT
    )
    self.assertIn("config_updated", self._drain_event_names())

  def test_simple_model_lines_still_parsed(self):
    self._feed("W1026 elastic_workload.py:393] >>> [WORKLOAD READY] Waiting for"
               " Start signal... <<<", 1000.0)
    self.assertEqual(self.state.jobset_status, "READY")
    self._feed("W1026 >>> [CONTROL] TRAINING STARTED signal received! <<<",
               1001.0)
    self.assertEqual(self.state.jobset_status, "RUNNING")
    self._feed("I1026 elastic_workload.py:529] Step 5 | Active Slices: 2 |"
               " Loss: 7.1234 | Step Time: 0.6000s", 1002.0)
    self.assertEqual(self.state.current_step, 5)
    self.assertAlmostEqual(self.state.last_loss, 7.1234)
    # simple_model also logs Pathways slice checks; they must not be treated
    # as MaxText recovery stages.
    self._feed("I1026 elastic.py:223] Sufficient slices active: 2 >= 1. Active"
               " indices: frozenset({0, 1})", 1003.0)
    self.assertEqual(self.state.jobset_status, "RUNNING")
    self.assertEqual(
        self.state.get_config()["workload_type"], bridge.WORKLOAD_SIMPLE_MODEL
    )

  def test_log_replay_suppresses_line_events(self):
    self.state.replaying_logs = True
    self.state.record_log("historic line", 1.0)
    self._feed(_maxtext_step(3), 1000.0)
    self.assertEqual(
        [n for n in self._drain_event_names()], ["config_updated"]
    )
    self.assertEqual(self.state.current_step, 3)
    self.tailer._finish_log_replay()
    self.assertFalse(self.state.replaying_logs)
    self.assertEqual(self._drain_event_names(), ["init"])

  def test_snapshot_exposes_workload_capabilities(self):
    self.state.update_config({"workload_type": bridge.WORKLOAD_MAXTEXT})
    snap = self.state.get_snapshot()
    self.assertEqual(snap["workload_type"], bridge.WORKLOAD_MAXTEXT)
    self.assertFalse(snap["capabilities"]["training_controls"])
    self.assertFalse(snap["capabilities"]["workload_rpc"])
    self.assertFalse(snap["capabilities"]["mode_switching"])
    self.assertTrue(snap["capabilities"]["k8s_disruption"])


class MaxTextTemplateTest(parameterized.TestCase):
  """The committed MaxText template renders into a deployable JobSet."""

  @parameterized.named_parameters(
      ("pause_resume", "-1", "step_snapshot_pause_resume"),
      ("replica_resize", "1", "step_snapshot_replica_resize"),
  )
  def test_template_renders_to_maxtext_jobset(self, min_slices,
                                              expected_mode):
    values = {
        "JOBSET_NAME": "maxtext-demo",
        "RUN_NAME": "maxtext-demo-run",
        "NAMESPACE": "ml",
        "KUEUE_QUEUE": "multislice-queue",
        "CPU_NODEPOOL": "cpu-pool",
        "NUM_SLICES": "2",
        "TPU_INSTANCE_TYPE": "tpuv6e:4x4",
        "TPU_ACCELERATOR": "tpu-v6e-slice",
        "TPU_TOPOLOGY": "4x4",
        "HOSTS_PER_SLICE": "4",
        "CHIPS_PER_HOST": "4",
        "GCS_OUTPUT": "gs://bucket/maxtext",
        "DATASET_FILES": "gs://bucket/c4/c4-train.array_record*",
        "ELASTIC_MIN_SLICE_COUNT": min_slices,
        "STEPS": "500",
        "MAXTEXT_IMAGE": "example.com/maxtext:tag",
        "PATHWAYS_SERVER_IMAGE": "example.com/server:tag",
        "PATHWAYS_PROXY_IMAGE": "example.com/proxy:tag",
        "COLOCATED_PYTHON_IMAGE": "example.com/sidecar:tag",
    }
    with open(_MAXTEXT_TEMPLATE_PATH, encoding="utf-8") as f:
      text = f.read()
    self.assertEqual(set(re.findall(r"\{\{([A-Z_]+)\}\}", text)), set(values))
    for key, value in values.items():
      text = text.replace("{{%s}}" % key, value)
    self.assertNotIn("{{", text)

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "elastic_demo_maxtext.yaml")
      with open(path, "w", encoding="utf-8") as f:
        f.write(text)
      inferred = bridge._infer_config_from_yaml(path)
    self.assertEqual(inferred["workload_type"], bridge.WORKLOAD_MAXTEXT)
    self.assertEqual(inferred["jobset_name"], "maxtext-demo")
    self.assertEqual(inferred["namespace"], "ml")
    self.assertEqual(inferred["num_slices"], 2)
    self.assertEqual(inferred["tpu_type"], "tpuv6e")
    self.assertEqual(inferred["topology"], "4x4")
    self.assertEqual(inferred["total_steps"], 500)
    self.assertEqual(inferred["elasticity_mode"], expected_mode)

    doc = yaml.safe_load(text)
    self.assertEqual(
        doc["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
        "multislice-queue",
    )
    worker = [rj for rj in doc["spec"]["replicatedJobs"]
              if rj["name"] == "pathways-worker"][0]
    self.assertEqual(worker["template"]["spec"]["completions"], 4)
    self.assertEqual(worker["template"]["spec"]["parallelism"], 4)
    self.assertIn('RUN_NAME="maxtext-demo-run"', text)


class ControlsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("maxtext_start", bridge.WORKLOAD_MAXTEXT, "start", "k8s", True),
      ("maxtext_pause", bridge.WORKLOAD_MAXTEXT, "pause", "k8s", True),
      ("maxtext_reset", bridge.WORKLOAD_MAXTEXT, "reset", "k8s", True),
      ("maxtext_rpc", bridge.WORKLOAD_MAXTEXT, "interrupt", "workload", True),
      ("maxtext_k8s", bridge.WORKLOAD_MAXTEXT, "interrupt", "k8s", False),
      ("maxtext_uncordon", bridge.WORKLOAD_MAXTEXT, "reconnect", "k8s", False),
      ("maxtext_mode", bridge.WORKLOAD_MAXTEXT, "mode", "k8s", True),
      ("simple_start", bridge.WORKLOAD_SIMPLE_MODEL, "start", "k8s", False),
      ("simple_rpc", bridge.WORKLOAD_SIMPLE_MODEL, "interrupt", "workload",
       False),
      ("simple_mode", bridge.WORKLOAD_SIMPLE_MODEL, "mode", "k8s", False),
  )
  def test_unsupported_action_message(self, workload, action, mechanism,
                                      rejected):
    message = bridge._unsupported_action_message(workload, action, mechanism)
    self.assertEqual(bool(message), rejected)

  @parameterized.named_parameters(
      ("maxtext_keeps_configmap", bridge.WORKLOAD_MAXTEXT, ["jobset"]),
      ("simple_model_deletes_configmap", bridge.WORKLOAD_SIMPLE_MODEL,
       ["jobset", "configmap"]),
  )
  def test_delete_workload(self, workload, expected_kinds):
    cfg = {
        "cluster_context": "test-context",
        "namespace": "default",
        "jobset_name": "demo",
        "workload_type": workload,
    }
    with mock.patch.object(
        bridge.subprocess,
        "run",
        return_value=subprocess.CompletedProcess([], 0, "deleted", ""),
    ) as run:
      ok, _ = bridge._delete_workload(cfg)
    self.assertTrue(ok)
    kinds = [call.args[0][call.args[0].index("delete") + 1]
             for call in run.call_args_list]
    self.assertEqual(kinds, expected_kinds)


class DeleteCommandTest(parameterized.TestCase):
  """`--delete` must not remove the shared ConfigMap for unknown workloads."""

  def setUp(self):
    super().setUp()
    with mock.patch.object(
        bridge, "_current_kubeconfig_context", return_value="test-context"
    ):
      fresh_state = bridge.BridgeState()
    self.enter_context(mock.patch.object(bridge, "state", fresh_state))
    self.run = self.enter_context(
        mock.patch.object(
            bridge.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0, "deleted", ""),
        )
    )

  @parameterized.named_parameters(
      ("jobset_gone_auto", None, [], ["jobset"]),
      ("jobset_gone_simple_model_flag", None, ["--workload=simple_model"],
       ["jobset", "configmap"]),
      ("live_simple_model", _SIMPLE_MODEL_COMMAND, [],
       ["jobset", "configmap"]),
      ("live_maxtext", _MAXTEXT_COMMAND.format(min_slices=-1), [],
       ["jobset"]),
  )
  def test_delete(self, live_command, extra_flags, expected_kinds):
    live_doc = _jobset(live_command) if live_command else None
    with mock.patch.object(
        bridge, "_fetch_live_jobset", return_value=live_doc
    ), self.assertRaises(SystemExit) as exit_ctx:
      bridge.main([
          "--context=test-context",
          "--jobset-name=maxtext-demo",
          "--namespace=default",
          "--delete",
          *extra_flags,
      ])
    self.assertEqual(exit_ctx.exception.code, 0)
    kinds = [call.args[0][call.args[0].index("delete") + 1]
             for call in self.run.call_args_list]
    self.assertEqual(kinds, expected_kinds)

  def test_delete_workload_can_skip_configmap(self):
    cfg = {
        "cluster_context": "test-context",
        "namespace": "default",
        "jobset_name": "demo",
        "workload_type": bridge.WORKLOAD_SIMPLE_MODEL,
    }
    ok, _ = bridge._delete_workload(cfg, delete_configmap=False)
    self.assertTrue(ok)
    self.assertLen(self.run.call_args_list, 1)
    self.assertIn("jobset", self.run.call_args_list[0].args[0])


if __name__ == "__main__":
  absltest.main()

