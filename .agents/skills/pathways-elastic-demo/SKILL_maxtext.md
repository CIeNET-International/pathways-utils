---
name: pathways-elastic-maxtext-demo
description: >-
  Guides users end-to-end through customizing, deploying, monitoring, and
  elasticity-testing the MaxText Llama 3.1 8B variant of the Pathways elastic
  training GKE demo defined by elastic_demo_maxtext.yaml in pathways-utils. Use
  when a user wants to run MaxText elastic training (in-memory snapshot backup
  with pause/resume or replica resize) on GKE TPU slices through Pathways, adapt
  elastic_demo_maxtext.yaml to their cluster (Kueue queues, CPU node pools, TPU
  type and topology, GCS buckets, dataset location, image registries), inject
  slice preemptions with kubectl cordon and drain, or verify elastic recovery
  from MaxText logs. Don't use for the toy elastic_workload.py demo rendered by
  gke_demo_bridge (see SKILL.md in this folder) or for MaxText runs without
  Pathways.
---

# Pathways Elastic Training GKE Demo: MaxText Variant

The MaxText variant runs a single-document Pathways `JobSet` with MaxText pre-training (`maxtext.trainers.pre_train.train`, `model_name=llama3.1-8b`, C4 through Grain's `ElasticIterator`) as a single-controller job with `elastic_enabled=True`, in-memory snapshot backups, and colocated Python checkpointing. Every user renders their own `elastic_demo_maxtext.yaml` (repository root, git-ignored) from the committed template [elastic_demo_maxtext.template.yaml](../../../pathwaysutils/extras/elastic/elastic_demo_maxtext.template.yaml) by filling in its `{{PLACEHOLDER}}` values (Step 2). Unlike the toy demo in [SKILL.md](SKILL.md), it has no `ConfigMap`, no `--render-only` generator, and no port-`8090` control server. Monitoring and fault injection work with plain `kubectl` or with the `gke_demo_bridge` dashboard, which recognizes MaxText JobSets (see Step 4).

## End-to-End Flow

1. Inspect local tools and the target cluster read-only, then ask one grouped set of questions (Step 1).
2. Render `elastic_demo_maxtext.yaml` from the template for the cluster, slice shape, storage, images, and elasticity mode (Step 2).
3. Validate with a server-side dry run and preflight checks (Step 3).
4. Deploy, confirm Kueue admission, and monitor training from the head pod logs (Step 4).
5. Inject a slice disruption and verify elastic recovery (Step 5).
6. Delete the `JobSet`, confirm no TPU nodes stay cordoned, and decide what to keep in GCS (Step 6).

---

## Step 1: Inspect the Environment & Ask Upfront Questions

Run the read-only checks from [SKILL.md](SKILL.md) Steps 1 and 3: `gcloud` project, `kubectl` contexts, node pools with accelerator/topology labels and taints, cordoned nodes, active pods per TPU pool, the `jobsets.jobset.x-k8s.io` CRD, and Kueue `localqueues`. Record `{context}` and pass `--context={context}` on every `kubectl` command. Only `kubectl` and `gcloud` are needed to deploy this variant; the `.venv` from SKILL.md Step 2 is only needed for the slice-shape helper (Step 2) and the optional dashboard (Step 4).

Also check, read-only:
- **Image access.** The template header lists the images of the validated run: the Pathways runtime (`unsanitized_server`, `unsanitized_proxy_server`) and the colocated Python sidecar from the internal `us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/` registry, and a MaxText image with elastic snapshot support from `gcr.io/cloud-tpu-multipod-dev/`. Confirm each tag exists with `gcloud artifacts docker images describe {image}`, and confirm the cluster's nodes can pull from those registries (for example, other pods on the cluster already run images from them). Otherwise ask for accessible equivalents from one matching runtime build.
- **GCS identity.** `kubectl --context={context} get sa default -n {namespace} -o jsonpath='{.metadata.annotations}'` shows the Workload Identity service account the head pod and workers use. It needs read access to the dataset bucket and read/write access to the output bucket.
- **Capacity.** Count idle TPU node pools matching the target accelerator and topology; each slice needs one whole pool.

Never guess clusters, buckets, queues, or images. Ask one concise, grouped set of questions covering whatever is still unknown:
1. Project, cluster, region or zone, `{context}`, namespace, Kueue `LocalQueue`, and the CPU node pool for `pathways-head`.
2. TPU type, topology, and slice count. The verified baseline is 2 x `tpuv5e` `4x8`. Llama 3.1 8B at `per_device_batch_size=1` and `max_target_length=8192` with in-slice FSDP fits a 32-chip v5e slice; for smaller slices, ask whether to lower `max_target_length` or change `model_name`.
3. Elasticity mode, set through `elastic_min_slice_count`:
   - `-1` (default, verified): snapshot **pause/resume**. Every attempt waits for all slices. When a slice is lost, training pauses until the replacement worker rejoins, then restores the newest in-memory snapshot or checkpoint and reuses the compiled step functions.
   - A value below the slice count (for example `1`): **replica resize**. Training continues on the surviving slices and scales back up when the slice returns; MaxText rebuilds the mesh and recompiles for each slice set. It relies on `grain_use_elastic_iterator=true`, which the manifest already sets.
4. GCS output location (`GCS_OUTPUT`, also used as the Pathways `--gcs_scratch_location`) and `RUN_NAME`. Each Llama 3.1 8B + AdamW checkpoint is about 54 GiB, and reusing an existing `RUN_NAME` resumes from its latest complete checkpoint instead of starting fresh.
5. Dataset location. Keep `DATASET_FILES` on the same continent as the cluster: `gs://maxtext-dataset/array-record/c4/en/3.0.1/c4-train.array_record*` (US) or `gs://max-datasets-rogue-europe-west4/array-record/c4/en/3.0.1/c4-train.array_record*` (EU).
6. Run length (`STEPS`, default `3000`, about 3.4 hours at 4 s/step); the template keeps checkpoints every 20 steps and retains the newest 3.
7. Images: the MaxText image (needs elastic snapshot support), Pathways server, Pathways proxy, and colocated Python sidecar. Default to the validated images in the template header when the cluster can pull them, and keep the server, proxy, and sidecar from the same runtime build.

---

## Step 2: Render `elastic_demo_maxtext.yaml` from the Template

Fill in every placeholder of [elastic_demo_maxtext.template.yaml](../../../pathwaysutils/extras/elastic/elastic_demo_maxtext.template.yaml) and write the result to `elastic_demo_maxtext.yaml` at the repository root (git-ignored). Leave the MaxText performance flags, the worker restart wrapper (always exits `1` so the kubelet restarts the Pathways worker), and the resource-manager port-rotation wrappers unchanged unless the user asks. To keep several variants, such as pause/resume and replica resize, render each to its own `elastic_demo_maxtext*.yaml` file with a distinct JobSet and run name.

| Placeholder | Value |
|---|---|
| `JOBSET_NAME` | JobSet name, at most 22 characters: the JobSet webhook rejects names whose coordinator label `{name}-pathways-head-0-0.{name}` exceeds 63 bytes. |
| `RUN_NAME` | MaxText run directory under `GCS_OUTPUT`, usually the JobSet name. Reusing one resumes from its latest complete checkpoint. |
| `NAMESPACE`, `KUEUE_QUEUE` | Namespace and Kueue `LocalQueue`. Without Kueue, delete the `labels:` and `kueue.x-k8s.io/queue-name` lines. |
| `CPU_NODEPOOL` | CPU node pool for `pathways-head`; keep its container `requests`/`limits` within one node's allocatable CPU and memory. |
| `NUM_SLICES` | Slice count; fills the worker `replicas`, `pathways-rm --instance_count`, and `pathways-proxy --num_elastic_slices`. |
| `TPU_INSTANCE_TYPE`, `TPU_ACCELERATOR`, `TPU_TOPOLOGY`, `HOSTS_PER_SLICE`, `CHIPS_PER_HOST` | Slice shape, derived with the helper below. |
| `GCS_OUTPUT`, `DATASET_FILES` | Output and scratch path, and the Grain ArrayRecord glob (Step 1). |
| `ELASTIC_MIN_SLICE_COUNT` | `-1` for pause/resume; any value below `NUM_SLICES` (for example `1`) for replica resize. |
| `STEPS` | Training steps. |
| `MAXTEXT_IMAGE`, `PATHWAYS_SERVER_IMAGE`, `PATHWAYS_PROXY_IMAGE`, `COLOCATED_PYTHON_IMAGE` | Images (Step 1). |

Other knobs stay as written in the template body: `elastic_snapshot_interval` (`5`), `elastic_timeout_seconds` (`3600`, how long an attempt waits for slices), `checkpoint_period` (`20`), `max_num_checkpoints_to_keep` (`3`), the TPU `tolerations` (`google.com/tpu=present:NoSchedule`), and the model flags.

Derive the slice-shape values instead of guessing by rendering the toy demo for the same shape into a scratch file (works offline; the dummy values are never deployed). Map `--instance_type` to `TPU_INSTANCE_TYPE`, `gke-tpu-accelerator` to `TPU_ACCELERATOR`, `gke-tpu-topology` to `TPU_TOPOLOGY`, the worker `completions` to `HOSTS_PER_SLICE`, and the worker `google.com/tpu` limit to `CHIPS_PER_HOST`:

```bash
.venv/bin/python -m pathwaysutils.extras.elastic.gke_demo_bridge \
  --context=dummy --image=unused --gcs-scratch=gs://unused/ \
  --tpu-type={tpu_type} --topology={topology} --num-slices={num_slices} \
  --render-only /tmp/shape.yaml
grep -nE '^ *(replicas|completions|parallelism|google\.com/tpu|cloud\.google\.com/gke-tpu-(accelerator|topology)): |--instance_(type|count)=' /tmp/shape.yaml
```

For example, `tpuv5e` `4x8` gives 8 hosts x 4 chips on `tpu-v5-lite-podslice`, `tpuv6e` `4x4` gives 4 hosts x 4 chips on `tpu-v6e-slice`, and `tpuv5e` `16x16` gives 64 hosts x 4 chips.

Render with `sed`. Values must not contain `|` or `&`; without Kueue, also pass `-e '/^  labels:$/d' -e '/kueue.x-k8s.io\/queue-name/d'`:

```bash
sed \
  -e 's|{{JOBSET_NAME}}|{jobset_name}|g' \
  -e 's|{{RUN_NAME}}|{run_name}|g' \
  -e 's|{{NAMESPACE}}|{namespace}|g' \
  -e 's|{{KUEUE_QUEUE}}|{local_queue}|g' \
  -e 's|{{CPU_NODEPOOL}}|{cpu_nodepool}|g' \
  -e 's|{{NUM_SLICES}}|{num_slices}|g' \
  -e 's|{{TPU_INSTANCE_TYPE}}|{instance_type}|g' \
  -e 's|{{TPU_ACCELERATOR}}|{accelerator}|g' \
  -e 's|{{TPU_TOPOLOGY}}|{topology}|g' \
  -e 's|{{HOSTS_PER_SLICE}}|{hosts_per_slice}|g' \
  -e 's|{{CHIPS_PER_HOST}}|{chips_per_host}|g' \
  -e 's|{{GCS_OUTPUT}}|{gcs_output}|g' \
  -e 's|{{DATASET_FILES}}|{dataset_files}|g' \
  -e 's|{{ELASTIC_MIN_SLICE_COUNT}}|{elastic_min_slice_count}|g' \
  -e 's|{{STEPS}}|{steps}|g' \
  -e 's|{{MAXTEXT_IMAGE}}|{maxtext_image}|g' \
  -e 's|{{PATHWAYS_SERVER_IMAGE}}|{server_image}|g' \
  -e 's|{{PATHWAYS_PROXY_IMAGE}}|{proxy_image}|g' \
  -e 's|{{COLOCATED_PYTHON_IMAGE}}|{sidecar_image}|g' \
  pathwaysutils/extras/elastic/elastic_demo_maxtext.template.yaml > elastic_demo_maxtext.yaml
grep -nE '[{]{2}' elastic_demo_maxtext.yaml   # must print nothing
```

---

## Step 3: Validate & Preflight

1. Validate against the API server. This catches schema errors, the coordinator-label length limit, and Kueue webhook problems; Kueue setting `suspend: true` in the dry-run output is expected.
   ```bash
   kubectl --context={context} apply --dry-run=server -f elastic_demo_maxtext.yaml
   ```
2. Confirm the run directory is empty unless the user wants to resume: `gcloud storage ls {GCS_OUTPUT}/{RUN_NAME}/` should report no objects.
3. Re-check that enough idle TPU node pools exist and that no TPU nodes are left cordoned from earlier runs (offer `kubectl --context={context} uncordon {node_name}`).
4. Optionally verify that every MaxText override key in the `jax-tpu` command exists in `src/maxtext/configs/base.yml` of the MaxText revision baked into the image; unknown keys fail config validation only after the 150-second startup wait.

---

## Step 4: Deploy & Monitor

> [!WARNING]
> `JobSet` templates are immutable. Delete an existing `JobSet` with the same name (Step 6) before applying an edited manifest.

```bash
kubectl --context={context} apply -f elastic_demo_maxtext.yaml
kubectl --context={context} get workloads -n {namespace}      # Kueue admission (ADMITTED=True)
kubectl --context={context} get pods -n {namespace} -o wide \
  -l jobset.sigs.k8s.io/jobset-name={jobset_name}
```

Follow training from the head pod:

```bash
HEAD=$(kubectl --context={context} -n {namespace} get pods \
  -l jobset.sigs.k8s.io/jobset-name={jobset_name},jobset.sigs.k8s.io/replicatedjob-name=pathways-head \
  -o jsonpath='{.items[0].metadata.name}')
kubectl --context={context} -n {namespace} logs -f "$HEAD" -c jax-tpu \
  | grep -E "completed step|snapshot|save checkpoint|Elastic|Restor|Traceback"
```

What a healthy run looks like (verified on 2 x v5e `4x8`):
- Workers pull the sidecar (about 1 GB) and server images in under 30 seconds, and `pathways-rm` logs `*** 2/2 Pathways Slices Now Ready ***`.
- The `jax-tpu` container sleeps 150 seconds, initializes, compiles, and logs `completed step: 0` about 7 minutes after `kubectl apply`.
- Steady state: `completed step: N, seconds: ~4.05, live slice count: 2, TFLOP/s/device: ~104, Tokens/s/device: ~2020, ... loss: ...`, with loss falling from about 12.3 (below 7 by step 150).
- `Saving in-memory snapshot at step N` every `elastic_snapshot_interval` steps, except `Skipping the in-memory snapshot at step N because a checkpoint save is in progress`.
- `Started save checkpoint @ {GCS_OUTPUT}/{RUN_NAME}/checkpoints/N` every `checkpoint_period` steps. Steps next to a save can report distorted `seconds` and `TFLOP/s` values because of asynchronous timing.
- Benign noise: goodput `Error in rolling window ... 'NoneType' object cannot be interpreted as an integer` and `Memstats unavailable ... only supported for addressable PjRt devices`.

### Monitoring Dashboard (Optional)

`gke_demo_bridge` recognizes MaxText JobSets and serves the same single-page dashboard on port `48259`. Use the `.venv` from SKILL.md Step 2, and add `--no-browser` plus SSH port forwarding on remote VMs as in SKILL.md Step 6.

```bash
# Deploy the manifest and monitor it (name and namespace come from the YAML):
.venv/bin/python -m pathwaysutils.extras.elastic.gke_demo_bridge \
  --context={context} --yaml=elastic_demo_maxtext.yaml
# Or attach to an already running JobSet without re-applying anything:
.venv/bin/python -m pathwaysutils.extras.elastic.gke_demo_bridge \
  --context={context} --jobset-name={jobset_name} --namespace={namespace}
```

- At startup the bridge logs `Workload type: maxtext (elasticity mode step_snapshot_pause_resume)`. When it attaches to a running pod, it replays the earlier log history with the original timestamps before streaming live updates.
- The dashboard charts step, loss, goodput, and live slice count from the `completed step` lines, and steps through the recovery stages during Step 5. Goodput only credits steps beyond the furthest step reached, so recomputed steps count as lost time.
- `Start`, `Pause`, and `Reset` are hidden, `In-Workload RPC` is disabled, and the elasticity-mode selectors are read-only, because MaxText has no control server and takes its mode from `elastic_min_slice_count`; the bridge answers those API calls with HTTP 409. `Node Cordon + Drain` and `Reconnect Slice` work.
- `Delete` removes only the `JobSet`. `Delete` and `Reconnect Slice` uncordon every cordoned TPU node in the cluster, so avoid them on shared clusters where others cordon nodes for maintenance.

---

## Step 5: Inject a Slice Disruption & Verify Recovery

Confirm with the user before disrupting anything, and only touch nodes that host this `JobSet`'s worker pods. With the dashboard open (Step 4), `Interrupt Slice {n}` with `Node Cordon + Drain` performs steps 1-2 on a non-leader worker of that slice, and `Reconnect Slice` performs step 4; the `kubectl` commands below do the same without the dashboard.

1. Pick a non-leader worker (completion index `1` or higher) of one slice. Draining a slice's index-`0` pod can move the slice's exclusive-placement anchor to a different node pool.
   ```bash
   TARGET=$(kubectl --context={context} -n {namespace} get pods \
     -l jobset.sigs.k8s.io/jobset-name={jobset_name},jobset.sigs.k8s.io/replicatedjob-name=pathways-worker,jobset.sigs.k8s.io/job-index=1,batch.kubernetes.io/job-completion-index=3 \
     -o jsonpath='{.items[0].metadata.name}')
   NODE=$(kubectl --context={context} -n {namespace} get pod "$TARGET" -o jsonpath='{.spec.nodeName}')
   ```
2. Note the last `completed step`, then cordon the node and evict only this `JobSet`'s pod from it:
   ```bash
   kubectl --context={context} cordon "$NODE"
   kubectl --context={context} drain "$NODE" --pod-selector=jobset.sigs.k8s.io/jobset-name={jobset_name} \
     --ignore-daemonsets --delete-emptydir-data --force --grace-period=0 --timeout=120s
   ```
3. Expect this sequence while the node stays cordoned (pause/resume, verified):
   - `completed step` lines stop. About 15 seconds after the drain the head logs `DATA_LOSS: RPC cancelled due to ... {node} exiting ... lost connection to peer`, then `Elastic event detected`.
   - MaxText cleans up: `Elastic utils: Checking for incomplete checkpoint after an elastic event...`. If a checkpoint save was in flight, it logs `No commit_success file found. Deleting {GCS_OUTPUT}/{RUN_NAME}/checkpoints/{step}/...`; that checkpoint is saved again after recovery.
   - `Elastic attempt 2` and `Waiting for 2 slices. Poll interval: 10, Timeout: 3600`, followed every 10 seconds by `1 slices active. Wanting at least 2. Active indices: frozenset({...})`. The interleaved `check_value_ready: DATA_LOSS` warnings come from slice health probes and are harmless. Pathways slice indices in these lines need not match the JobSet `job-index`.
   - The Job controller creates a replacement worker pod that stays `Pending`: exclusive placement pins it to the slice's node pool, and the cordoned node is the only free node there. The other workers of that slice restart once; workers in the healthy slice do not restart.
   - Training waits up to `elastic_timeout_seconds` for the slice to return.
4. Uncordon to simulate repaired capacity:
   ```bash
   kubectl --context={context} uncordon "$NODE"
   ```
5. Expect recovery (verified timings on 2 x v5e `4x8`):
   - About 30 seconds after uncordon the replacement worker is running, `pathways-rm` has reallocated the slice, and the head logs `Sufficient slices active: 2 >= 2`.
   - Within seconds: `Restoring the in-memory snapshot at step N...`, `Restored the data iterator state: {'global_next_index': ...}`, and `Restored the in-memory snapshot at step N.` MaxText restores the snapshot when it is at least as recent as the latest complete checkpoint, otherwise the checkpoint. Steps after `N` (at most `elastic_snapshot_interval`) are recomputed.
   - The first post-recovery `completed step: N+1, ... live slice count: 2` arrives about 2 minutes after uncordon (its `seconds` value includes the re-setup), and step time returns to about 4 seconds.
   - Verified run: drained right after step 160 while the step-160 checkpoint was saving; the incomplete checkpoint was deleted, the step-155 snapshot was restored, and `completed step: 156` appeared 135 seconds after uncordon. Excluding the deliberate two-minute cordon, the disruption cost about 14 seconds of detection, 135 seconds of recovery, and 5 recomputed steps.
6. Run the Step 6 check to confirm no TPU nodes remain cordoned.

With **replica resize** (`elastic_min_slice_count` below the slice count), training continues instead of pausing (verified on 2 x v5e `4x8`):
- About 20 seconds after the drain, `Elastic event detected` is followed at once by `Elastic attempt 2` and `Sufficient slices active: 1 >= 1`. MaxText restores the newest backup onto the surviving slice (`Restoring the checkpoint at step N rather than the in-memory snapshot at step M, which is older.` when the checkpoint is newer) and recompiles for one slice. The first `completed step ... live slice count: 1` arrived about 2 minutes 20 seconds after detection, at half the global batch, and the dashboard shows `Degraded (1/2)`.
- The verified run took about 14.2 s per step on one slice versus about 4 s on two (about 30 versus 104 TFLOP/s/device) even though per-chip work is unchanged; report degraded-mode throughput as an open issue rather than expected behavior.
- After uncordon, the Pathways monitor logs `Newly available but inactive slices frozenset({...})`, MaxText logs `... a new slice is available ...` (it waits for any checkpoint save in progress), and the manager logs `Scale up requested.`, about 1 minute 50 seconds after uncordon in the verified run. MaxText restores the newest backup on both slices, recompiles, and resumes with `live slice count: 2` about 2 minutes later, recomputing the steps after that backup.

---

## Step 6: Clean Up

```bash
kubectl --context={context} -n {namespace} delete jobset {jobset_name}
kubectl --context={context} get nodes --selector=cloud.google.com/gke-tpu-accelerator \
  -o jsonpath='{.items[?(@.spec.unschedulable==true)].metadata.name}'
```

Uncordon any node the test cordoned. Then ask whether to keep or delete `{GCS_OUTPUT}/{RUN_NAME}/` (checkpoints at about 54 GiB each, TensorBoard metrics, profiles). Delete it only with explicit confirmation, for example `gcloud storage rm -r {GCS_OUTPUT}/{RUN_NAME}/`. Deleting it, or choosing a new `RUN_NAME`, is also how to start the next run from step 0.

