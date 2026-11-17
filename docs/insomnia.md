# Running vTSFM on Columbia's Insomnia cluster

Insomnia's GPU nodes can run all four adapters, but the cluster differs from the
A100 reference machine in four ways that each break a naive setup:

1. **The NVIDIA driver is too old for CUDA 13.** Nodes run driver 575.57.08
   (CUDA 12.9) or 550.54.14 (CUDA 12.4). The TimesFM, Toto, and Moirai stacks
   use `torch==2.14.0+cu130`, which needs driver 580 or newer and otherwise fails
   with "The NVIDIA driver on your system is too old". The fix is NVIDIA's CUDA
   forward-compatibility libraries, described below; no admin rights needed.
2. **Home has a 50 GB quota.** vLLM, Triton, uv, and Hugging Face all cache under
   `~/.cache` by default. When the quota fills, vLLM's engine fails to start.
   Keep environments and caches on the node-local `/local` disk.
3. **Compute-node internet can be very slow.** Some nodes download at under
   1 MB/s, while login nodes get about 100 MB/s. Model checkpoints and roughly
   10 GB of wheels per adapter can stall until tests time out. Download on a login
   node and copy into the job.
4. **Jobs are ephemeral.** `burst` jobs can be preempted after about 30 minutes,
   and `/local` is wiped when a job ends. Use `-p short`, keep each run
   self-contained, and copy results to home before the job exits.

## Verified configuration

Verified 2026-11-14/15 on an RTX A6000 (driver 575.57.08) at commit `bd1eca2`:

| Adapter | Stack | `doctor` | GPU contract tests | `test-models.sh` integration |
| --- | --- | --- | --- | --- |
| chronos2 | torch 2.9.1+cu128, vLLM 0.16.0 | ready, no compat libs needed | n/a | synthetic smoke run OK |
| timesfm25 | torch 2.14.0+cu130, vLLM 0.33.0 | ready with compat libs | 178/178 | passed |
| moirai2 | torch 2.14.0+cu130, vLLM 0.33.0 | ready with compat libs | 178/178 | 6/6 passed |
| toto2 | torch 2.14.0+cu130, vLLM 0.33.0 | ready with compat libs | 178/178 | 6/6 passed |

The forward-compatibility package is officially supported only on data-center
GPUs, and so far it has been verified only on the A6000. Insomnia also has
L40, L40S, H100, and RTX PRO 6000 Blackwell (sm_120) nodes. Torch 2.14 ships
sm_120 kernels, but those nodes are untested here. Run the CUDA check below
whenever you land on a new GPU type.

## 1. Get a GPU

Use the course account on the `short` partition. It started within minutes in
testing and was not preempted:

```bash
srun -A edu -p short --gres=gpu:1 -c 4 --mem=32G -t 03:00:00 --pty bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
```

`burst` (`-p burst --qos=burst`) may start sooner but can be preempted after
about 30 minutes, losing everything on `/local`. If a node is noticeably slow,
add `--exclude=<node>`.

## 2. Put code, environments, and caches on `/local`

The cluster has no GitHub SSH key by default. Copy a real git clone, not a
`git archive`, because the integration tests call `git rev-parse`. For example,
from your laptop:

```bash
git bundle create vtsfm.bundle main
scp vtsfm.bundle <uni>@insomnia.rcs.columbia.edu:
```

Then on the compute node:

```bash
export R=/local/$USER-vtsfm-$SLURM_JOB_ID      # per-job dir: jobs on the same node don't collide
mkdir -p $R/tmp
git clone -q ~/vtsfm.bundle $R/vTSFM && cd $R/vTSFM && git checkout -q main

export UV_CACHE_DIR=$R/uv-cache HF_HOME=$R/hf TMPDIR=$R/tmp \
       XDG_CACHE_HOME=$R/xdg VLLM_CACHE_ROOT=$R/vllm-cache \
       TRITON_CACHE_DIR=$R/triton TORCHINDUCTOR_CACHE_DIR=$R/inductor \
       UV_HTTP_TIMEOUT=300
```

Use a per-job directory. When two jobs on the same node shared one directory,
the second job's `rm -rf` wiped the first job's environment.

## 3. Install the CUDA 13 forward-compatibility libraries

This step is needed for timesfm25, toto2, and moirai2, but not chronos2.

```bash
mkdir -p $R/compat && cd $R/compat
curl --retry 5 -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-compat-13-0-580.95.05-1.el9.x86_64.rpm
rpm2cpio cuda-compat-13-0-*.rpm | cpio -idm --quiet
export LD_LIBRARY_PATH=$R/compat/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd $R/vTSFM
```

Torch loads the user-space driver library `libcuda.so` through the dynamic loader.
With the compat folder first on `LD_LIBRARY_PATH`, it loads the 580-series
`libcuda.so`, which understands CUDA 13 and talks to the older kernel driver.
Nothing on the system changes, and the variable affects only processes started
from this shell. Set it before running `setup.sh`, `doctor.sh`, `serve.sh`, or
`test-models.sh`.

## 4. Build an environment

```bash
./scripts/setup.sh                      # chronos2 → .venv
./scripts/setup.sh --adapter timesfm25  # → .venv-timesfm25 (same for toto2, moirai2)
```

Build adapters one at a time. Two `setup.sh` runs in the same checkout fail with
"Could not acquire lock". On a node with good network a build takes 15–30
minutes; on a slow node it can take over an hour or time out (see step 6).

## 5. Check CUDA and run the tests

```bash
.venv-timesfm25/bin/python -c "import torch; x=torch.randn(512,512,device='cuda'); print(torch.cuda.get_device_name(0), float((x@x).sum()))"
./scripts/doctor.sh --adapter timesfm25          # expect "ready": true
.venv-timesfm25/bin/python -m pytest -q tests/test_cuda_contract.py tests/test_execution.py \
    tests/test_batching.py tests/test_result_transport.py
./scripts/test-models.sh --adapter timesfm25 --gpu 0 --port 18001 \
    --output ~/vtsfm-results/timesfm25-$(date +%Y%m%d-%H%M)
```

Point `--output` at home or copy the results there, because `/local` is deleted
when the job ends. For CPU tests, use `./scripts/setup.sh --cpu` and
`.venv/bin/python -m pytest -m 'not gpu'`.

## 6. If downloads are slow or stall

The symptom is a native reference that sits at `Fetching 2 files: 0%` until the
900-second test timeout, or `setup.sh` failing with "Failed to download
distribution due to network timeout". Do the downloads on a login node instead.

**Checkpoints.** Download them on the login node into home, which is small
(Toto 2 is 1.1 GB), then copy them into `$HF_HOME` inside the job:

```bash
# login node
HF_HOME=~/vtsfm-hf uvx --from 'huggingface_hub[cli]' hf download Datadog/Toto-2.0-313m \
    --revision a7bab288f5e95f8606f8306f86659357e1c001ef
# compute node
mkdir -p $HF_HOME && cp -r ~/vtsfm-hf/. $HF_HOME/
```

Model IDs and revisions live in `src/vtsfm/adapters/<adapter>/spec.py`.

**Wheels.** These are too big for home (about 9 GB per adapter). Fill a uv cache in
the login node's `/tmp`, then stream it into the job. The login node's `/tmp` is
`noexec`, so resolve without creating a venv there:

```bash
# login node, in a clone of the repo
export UV_CACHE_DIR=/tmp/$USER-uvc
uv pip install --python-version 3.12 --python-platform x86_64-manylinux_2_34 \
    --target /tmp/$USER-target --torch-backend cu130 ".[serve,dev,toto2]"
rm -rf /tmp/$USER-target
# still on the same login node: stream the cache into a job script that untars stdin
tar c -C /tmp/$USER-uvc . | srun -A edu -p short --gres=gpu:1 -c 4 --mem=32G -t 02:00:00 \
    bash -c 'R=/local/$USER-vtsfm-$SLURM_JOB_ID; mkdir -p $R/uv-cache && tar x -C $R/uv-cache && ...'
```

Use `--python-version 3.11` for timesfm25 and moirai2. For moirai2, also pass
`--override scripts/moirai2-overrides.txt`. With the cache in place, `setup.sh`
finishes in about 20 seconds. Insomnia has more than one login node, so run the
download and the `tar | srun` on the same one, check with `hostname`, and delete
the `/tmp` cache afterwards.
