# The Complete Slurm Guide: Scripting & Interactive Sessions

A practical, in-depth reference for using Slurm on HPC/GPU clusters — written with ML training workloads (PyTorch, multi-GPU, long-running jobs) specifically in mind, and customized for the **IIT Mandi DGX Cluster**.

> **Cluster quick facts (IIT Mandi DGX0 AI Cluster)**
> - Login: `ssh <username>@dgx-login.iitmandi.ac.in` — you never SSH into the DGX servers directly, everything goes through Slurm from the login node.
> - Check resource availability before submitting: `dgx-status`
> - Three GPU tiers, each its own partition — **full GPU**, **medium MIG**, **small MIG** (explained in detail in Section 7).
> - Currently only B200 partitions (`full-b200`, `medium-b200`, `small-b200`) are live; H200 partitions arrive later.
> - Access is granted via IT Helpdesk with faculty guide/chair approval — not self-service.
>
> Everywhere below where you see a generic `--partition=gpu` or `--gres=gpu:1` from standard Slurm docs, the equivalent **on this cluster** is one of the three tiers with cluster-specific GRES strings (e.g. `--gres=gpu:nvidia_b200:1` for a full GPU). Sections 4, 6, and 7 use the real syntax throughout.

---

## Table of Contents

1. [What Slurm Is and How It Thinks](#1-what-slurm-is-and-how-it-thinks)
2. [Core Concepts](#2-core-concepts)
3. [The Command-Line Toolkit](#3-the-command-line-toolkit)
4. [Writing `sbatch` Batch Scripts](#4-writing-sbatch-batch-scripts)
5. [Understanding Resource Requests Deeply](#5-understanding-resource-requests-deeply)
6. [Interactive Sessions: `srun` and `salloc`](#6-interactive-sessions-srun-and-salloc)
7. [GPU Jobs](#7-gpu-jobs)
8. [Multi-Node / Multi-GPU Distributed Training](#8-multi-node--multi-gpu-distributed-training)
9. [Job Arrays](#9-job-arrays)
10. [Job Dependencies & Pipelines](#10-job-dependencies--pipelines)
11. [Environments: Modules, Conda, and Virtualenvs](#11-environments-modules-conda-and-virtualenvs)
12. [Monitoring, Logs, and Debugging](#12-monitoring-logs-and-debugging)
13. [Checkpointing & Requeueing Long Jobs](#13-checkpointing--requeueing-long-jobs)
14. [Common Pitfalls](#14-common-pitfalls)
15. [Quick Reference Cheat Sheet](#15-quick-reference-cheat-sheet)

---

## 1. What Slurm Is and How It Thinks

**Slurm** (Simple Linux Utility for Resource Management) is a workload manager used on most HPC and GPU clusters. Its job is to answer one question fairly and efficiently for many competing users: *"Who gets to run what, on which machines, and for how long?"*

Three things Slurm does:

- **Allocates resources** — hands out nodes, CPUs, memory, and GPUs to jobs.
- **Schedules jobs** — decides the order jobs run in, based on priority, fairness, and resource availability.
- **Manages execution** — starts, monitors, and cleans up job processes across possibly many machines.

### Why this matters practically

You never run your training script directly on a cluster login node. Instead, you **describe what you need** (4 CPUs, 1 GPU, 32GB RAM, 6 hours) and Slurm finds you a machine (a "compute node") that has it, queues you if none is free yet, and runs your job there. This is true whether you're running one Python script or an interactive shell.

---

## 2. Core Concepts

| Term | Meaning |
|---|---|
| **Cluster** | The whole collection of machines managed by Slurm. |
| **Node** | A single physical/virtual machine (e.g., one server with 8 GPUs). |
| **Partition** | A named group of nodes with shared policies (e.g., `gpu`, `cpu`, `debug`). Similar to a "queue" in other schedulers. |
| **Job** | A resource allocation request, optionally with a script to run. Has a unique Job ID. |
| **Job step** | A unit of work *within* a job, launched via `srun`. A job can have multiple steps. |
| **Task** | An instance of your program. `--ntasks=4` means 4 copies of your program run (think MPI ranks). |
| **CPU / core** | A processing unit. `--cpus-per-task` controls threads available per task. |
| **QOS** | Quality of Service — a policy layer (priority, max walltime, max jobs) often stacked on top of partitions. |
| **Walltime** | Maximum real-world time your job is allowed to run before Slurm kills it. |

### The mental model

```
Cluster
 └── Partition "gpu"
      ├── Node gpu-a1  (8× B200, 128 cores, 1TB RAM)
      ├── Node gpu-a2  (8× B200, 128 cores, 1TB RAM)
      └── Node gpu-a3  ...

Your job → requests resources → Slurm scheduler queues/places it → runs on 1+ nodes
```

A **task** ≠ a **node** ≠ a **CPU**. Getting this distinction right is the single biggest source of confusion for newcomers — Section 5 covers it in detail.

---

## 3. The Command-Line Toolkit

| Command | Purpose |
|---|---|
| `dgx-status` | **IIT Mandi DGX-specific.** Cluster-provided wrapper showing current GPU/MIG availability across all three tiers at a glance — run this before submitting anything non-trivial. |
| `sinfo` | View cluster/partition state — what's available right now. |
| `squeue` | View the job queue — what's running/pending, yours or everyone's. |
| `sbatch script.sh` | Submit a batch script to run non-interactively. |
| `srun` | Run a command interactively/immediately, or as a step inside a job. |
| `salloc` | Allocate resources for an interactive session without immediately running a command. |
| `scancel <jobid>` | Cancel a job. |
| `scontrol show job <jobid>` | Detailed info about a specific job. |
| `sacct` | Historical accounting info — jobs that finished, failed, or were killed. |
| `sacctmgr` | Account/association management (usually admin-only, but useful for `show assoc` to check your allowed partitions/limits). |

### Quick tour

```bash
# IIT Mandi DGX: check GPU/MIG availability across all three tiers first
dgx-status

# What partitions exist, and their state? (standard Slurm view)
sinfo

# Example output on this cluster:
# PARTITION    AVAIL  TIMELIMIT  NODES  STATE  NODELIST
# full-b200    up     8:00:00    1      mix    dgx0
# medium-b200  up     4:00:00    1      mix    dgx0
# small-b200   up     2:00:00    1      idle   dgx0
# full-h200    down   -          -      -      (coming later)
# medium-h200  down   -          -      -      (coming later)
# small-h200   down   -          -      -      (coming later)

# What's currently queued/running? (just your jobs)
squeue --me
squeue -u $USER      # equivalent, used in the official cluster guide

# Everyone's jobs on the gpu partition
squeue -p gpu

# Detail on a specific job
scontrol show job 123456

# Cancel it
scancel 123456

# Your job history for today
sacct --starttime today --format=JobID,JobName,State,Elapsed,MaxRSS
```

`squeue` output field cheatsheet:

```
JOBID  PARTITION  NAME    USER    ST  TIME    NODES  NODELIST(REASON)
123456 gpu        train   mohit   R   2:15:03 1      gpu-a2
123457 gpu        eval    mohit   PD  0:00    1      (Priority)
```

`ST` = state: `R` running, `PD` pending, `CG` completing, `F` failed, `CD` completed, `TO` timeout, `CA` cancelled.

`REASON` (when pending) tells you *why* you're waiting — e.g. `(Resources)`, `(Priority)`, `(QOSMaxGRESPerUser)`. Always check this before assuming the cluster is "stuck."

---

## 4. Writing `sbatch` Batch Scripts

A batch script is a normal shell script with special `#SBATCH` comment directives at the top, which Slurm parses before running anything.

### Anatomy of a script

```bash
#!/bin/bash
#SBATCH --job-name=conformer_train        # Name shown in squeue
#SBATCH --partition=full-b200             # IIT Mandi DGX: full-b200 / medium-b200 / small-b200
#SBATCH --gres=gpu:nvidia_b200:1          # 1 full B200 GPU — see Section 7 for MIG variants
#SBATCH --cpus-per-task=16                # CPU cores per task
#SBATCH --mem=64G                         # Memory for the whole job
#SBATCH --time=08:00:00                   # Walltime: 8 hours (max is per-partition, check dgx-status)
#SBATCH --output=job_%j.log               # Matches the cluster's own log convention (job_<jobid>.log)

# ---- Everything below is a normal bash script ----
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load required modules here (check with `module avail` — the DGX cluster
# guide doesn't document a fixed module set, so confirm what's provisioned)
source ~/miniconda3/bin/activate patchconformer   # Activate your env

cd ~/projects/patch-conformer
python train.py --config configs/displace.yaml

echo "Job finished at $(date)"
```

Submit it with:

```bash
sbatch train_job.sh
```

You'll get back: `Submitted batch job 123456` — and your log will appear as `job_123456.log` in the directory you submitted from.

> **Note:** the official IIT Mandi guide's templates don't include `--nodes`/`--ntasks` (they default to 1 each) or `--mail-type`/`--mail-user`, and use a flat `job_%j.log` rather than a `logs/` subdirectory — so there's no directory to pre-create on this cluster, unlike the general-purpose advice below about the `--output` path needing to exist beforehand.

### Key points that trip people up

- **`#SBATCH` lines must come before any executable command** in the script (comments and blank lines are fine between them, but once bash logic starts, Slurm stops reading directives).
- The script runs on the **first allocated node** by default. If you need something on every node, you must explicitly `srun` it (see Section 6).
- **`%j`, `%x`, `%a`, `%N`** are useful filename patterns for `--output`/`--error`: job ID, job name, array index, node name respectively.
- You can override any `#SBATCH` directive from the command line at submit time — CLI flags win:
  ```bash
  sbatch --time=12:00:00 --gres=gpu:2 train_job.sh
  ```
- Make sure your `--output` directory **already exists** if you use a subdirectory path — Slurm won't create it and will silently fail to write logs (job may still run, but you'll get no output file). On the IIT Mandi DGX cluster the default convention is a flat `job_%j.log` in your submit directory, so this isn't an issue unless you customize the path yourself.

### Useful environment variables inside a job

| Variable | Meaning |
|---|---|
| `$SLURM_JOB_ID` | Current job's ID |
| `$SLURM_JOB_NAME` | Job name |
| `$SLURM_SUBMIT_DIR` | Directory you ran `sbatch` from |
| `$SLURM_NTASKS` | Total number of tasks |
| `$SLURM_CPUS_PER_TASK` | CPUs per task |
| `$SLURM_NODELIST` | Nodes allocated |
| `$SLURM_ARRAY_TASK_ID` | Index within a job array |
| `$SLURM_PROCID` | Rank of the current task (0-indexed) |
| `$SLURM_LOCALID` | Local rank on the current node |

Example — using the submit directory so paths work regardless of where Slurm actually schedules you:

```bash
cd $SLURM_SUBMIT_DIR
python train.py
```

---

## 5. Understanding Resource Requests Deeply

This is the part worth slowing down on, since it's the most common source of wasted compute or mysteriously slow jobs.

### Nodes vs. Tasks vs. CPUs-per-task

```
--nodes=2          # 2 machines
--ntasks=4          # 4 total processes across those nodes (e.g. 2 per node)
--cpus-per-task=8   # each process gets 8 CPU cores for its own threading
```

Total CPUs requested = `ntasks × cpus-per-task` = 32 cores across 2 nodes in this example.

- Use **`--ntasks`** when you're running multiple independent (or MPI-coordinated) processes — e.g. distributed training with one process per GPU.
- Use **`--cpus-per-task`** for how many threads/cores *each* process can use internally — e.g. PyTorch DataLoader workers, OpenMP threads, or intra-op parallelism.
- For a **single-GPU single-process job** (the common case for interactive debugging or a small training run):
  ```bash
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=8
  #SBATCH --gres=gpu:1
  ```

### Memory

```bash
#SBATCH --mem=64G           # Total memory for the whole job (all tasks on that node)
# OR
#SBATCH --mem-per-cpu=8G    # Memory per CPU core requested — total = cpus × this value
```

Never specify both — Slurm will reject the script. `--mem=0` on some clusters means "give me all memory on the node" (cluster-dependent; check local docs before relying on this).

### GPUs (`--gres` and `--gpus`)

Two common syntaxes, cluster-dependent — check which yours supports (`sinfo -o "%G"` shows configured GRES):

```bash
#SBATCH --gres=gpu:1              # 1 GPU, any type (generic — not what this cluster uses)
#SBATCH --gres=gpu:b200:2         # 2 GPUs, specifically B200s (if node has typed GPUs)

# Newer, more explicit alternative on some Slurm versions:
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=1
```

On the **IIT Mandi DGX cluster**, GPU type is baked into the GRES name itself (rather than a separate `sinfo`-discoverable type flag), and it also encodes whether you're getting a full GPU or a MIG slice — see the full table in Section 7:

```bash
#SBATCH --gres=gpu:nvidia_b200:1               # full B200
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1        # medium MIG slice (90GB)
#SBATCH --gres=gpu:nvidia_b200_1g.45gb:1        # small MIG slice (45GB)
```

Check what GPU types exist on a partition:

```bash
sinfo -p full-b200 -o "%N %G"
```

### Time limits

```bash
#SBATCH --time=2-00:00:00   # 2 days
#SBATCH --time=08:00:00     # 8 hours
#SBATCH --time=30:00        # 30 minutes
```

Request realistically — too-short a walltime gets your job killed mid-training; too-long a walltime can push you into lower scheduling priority or a different (slower-turnaround) QOS on some clusters. If unsure, check the partition's max with `sinfo` (the `TIMELIMIT` column), and use **checkpointing** (Section 13) so a timeout isn't catastrophic.

### Priority & QOS

```bash
#SBATCH --qos=normal          # Or "high", "long", "debug" — cluster-specific names
#SBATCH --partition=full-b200
```

Check what QOS options you have:

```bash
sacctmgr show assoc user=$USER format=account,partition,qos
```

The official IIT Mandi DGX guide doesn't document a separate QOS layer beyond the three partitions themselves — if `sacctmgr` shows something beyond the defaults, that's cluster-specific policy worth asking IT support about directly.

---

## 6. Interactive Sessions: `srun` and `salloc`

Interactive sessions are essential for debugging — e.g. stepping through a CUDA OOM error live, checking `nvidia-smi` on the actual node, or profiling a data loader before committing to a long batch job.

### `srun` — immediate, foreground

`srun` requests resources **and** immediately runs a command, blocking until it completes. Used two ways:

**A) Direct interactive shell** (most common for debugging):

On the IIT Mandi DGX cluster, the official user guide only documents batch (`sbatch`) submission — but `srun --pty bash` works the same way against the same partitions and GRES strings. For quick debugging, the **small MIG tier** is usually the right choice (fast to get allocated, plenty for sanity checks); reach for `full-b200` only when you actually need full-GPU memory/compute to reproduce a real issue.

```bash
# Quick debug shell on a small MIG slice
srun --partition=small-b200 --gres=gpu:nvidia_b200_1g.45gb:1 \
     --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash

# Heavier interactive debugging that needs a full GPU
srun --partition=full-b200 --gres=gpu:nvidia_b200:1 \
     --cpus-per-task=16 --mem=64G --time=02:00:00 --pty bash
```

- `--pty` gives you a proper interactive terminal (pseudo-tty) attached to the DGX compute node.
- You'll be dropped into a bash prompt **running on the compute node itself** — check with `hostname` and `nvidia-smi`.
- This blocks your terminal until the job ends (you exit, or walltime runs out) — the resources are held the whole time, so don't leave it idle, especially on `full-b200` where you're occupying a whole GPU other users may be queued for.
- Run `dgx-status` from another terminal (still on the login node) beforehand if you're unsure whether a tier is currently free.

```bash
$ srun --partition=small-b200 --gres=gpu:nvidia_b200_1g.45gb:1 --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
srun: job 123458 queued and waiting for resources
srun: job 123458 has been allocated resources
(dgx0) mohit@dgx0:~$ nvidia-smi
(dgx0) mohit@dgx0:~$ python -c "import torch; print(torch.cuda.is_available())"
```

**B) As a step inside an existing allocation** (see `salloc` below) — run one command at a time within resources you already hold.

### `salloc` — allocate first, decide what to run after

`salloc` reserves resources and drops you into a shell **on the login/head node**, from which you then `srun` individual commands into the allocation. This is more flexible when you want to launch several different commands (or multi-node/multi-process steps) without re-requesting resources each time.

```bash
salloc --partition=medium-b200 --gres=gpu:nvidia_b200_3g.90gb:1 \
       --cpus-per-task=8 --mem=64G --time=04:00:00

# Now inside the allocation shell (still on the login node, dgx-login.iitmandi.ac.in):
salloc: Granted job allocation 123459
$ srun nvidia-smi                     # runs on the allocated compute node
$ srun python check_env.py
$ exit                                # releases the allocation
```

(Since this cluster is a single DGX0 node sliced into full GPUs and MIG instances rather than a many-node HPC fleet, `--ntasks=2` multi-process steps within one `salloc` are less commonly needed here than on traditional multi-node clusters — see Section 8 for the multi-GPU distinction that actually applies.)

### `srun --pty bash` vs `salloc` — when to use which

| Use case | Tool |
|---|---|
| Quick one-off interactive debugging shell | `srun --pty bash` |
| Need to run several different commands/steps without re-queueing | `salloc`, then multiple `srun`s inside it |
| Testing multi-node or multi-process launch behavior | `salloc` + `srun --ntasks=N ...` |
| Running Jupyter/VS Code remote on a compute node | Either works — see below |

### Practical pattern: interactive GPU debugging session for your training pipeline

```bash
srun --partition=full-b200 --gres=gpu:nvidia_b200:1 --cpus-per-task=16 --mem=64G \
     --time=03:00:00 --job-name=debug_conformer --pty bash

# Once on the node:
source ~/miniconda3/bin/activate patchconformer
cd ~/projects/patch-conformer
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability())"
python train.py --config configs/debug.yaml --max_steps 20   # quick sanity run
```

This is the fastest way to catch things like a CUDA/PyTorch compute-capability mismatch (e.g. Blackwell/B200 needing CUDA 12.8+) *before* burning an 8-hour batch allocation on a crash at step 1.

### Interactive Jupyter on a compute node

```bash
# 1. Start an interactive allocation with a port reserved (small MIG is plenty for a notebook)
srun --partition=small-b200 --gres=gpu:nvidia_b200_1g.45gb:1 --cpus-per-task=8 --mem=32G \
     --time=04:00:00 --pty bash

# 2. On the compute node:
jupyter lab --no-browser --port=8888 --ip=0.0.0.0

# 3. From your local machine, tunnel through the DGX login node:
ssh -L 8888:dgx0:8888 <username>@dgx-login.iitmandi.ac.in
# then open http://localhost:8888 locally
```

(Replace `dgx0` with whatever node `hostname` reports once you're in the session, in case the cluster grows beyond a single DGX box.)

### Ending an interactive session

- `exit` or `Ctrl-D` in an `srun --pty bash` shell ends the job and releases resources.
- `exit` in a `salloc` shell releases the whole allocation.
- Don't just close your terminal without exiting cleanly if avoidable — orphaned allocations sit idle and burn your fair-share priority. If you do lose connection, `scancel <jobid>` from a fresh login afterward.

---

## 7. GPU Jobs

### The IIT Mandi DGX tier structure: full GPU vs. MIG

This is the part that's genuinely specific to this cluster and worth understanding properly. The DGX0 box's GPUs are exposed three ways using NVIDIA **MIG (Multi-Instance GPU)** partitioning — a single physical GPU can be sliced into smaller, hardware-isolated instances:

| Tier | Partition(s) | What you get | GRES string | Good for |
|---|---|---|---|---|
| **Full GPU** | `full-b200` (`full-h200` later) | An entire physical B200, all its memory and compute | `--gres=gpu:nvidia_b200:1` | Large model training, anything memory-hungry, multi-GPU jobs |
| **Medium MIG** | `medium-b200` (`medium-h200` later) | A 90GB MIG slice of a B200 (75GB for H200) | `--gres=gpu:nvidia_b200_3g.90gb:1` | Medium-scale training, inference on sizable models |
| **Small MIG** | `small-b200` (`small-h200` later) | A 45GB MIG slice of a B200 (35GB for H200) | `--gres=gpu:nvidia_b200_1g.45gb:1` | Development, quick tests, inference, notebooks |

Practically:
- A full B200 can be carved into **8 small MIG slices** or **4 medium MIG slices** and shared across users — that's why `small-b200`/`medium-b200` jobs tend to schedule faster than `full-b200` ones (more concurrent capacity, smaller footprint per job).
- **Use the smallest tier that fits your workload.** If your Conformer/RPCA experiment comfortably fits in 45GB, running it on `small-b200` gets you scheduled faster and leaves `full-b200` free for genuinely large jobs (yours or others').
- Only reach for `full-b200` when you need the full memory/compute of one GPU, or multiple GPUs at once (Section 8).
- Run `dgx-status` before choosing a tier — if `small-b200` is saturated but `medium-b200` is free, that's useful information before you submit and wait.

### Checking what you actually got

Never assume — verify inside the job:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo $CUDA_VISIBLE_DEVICES
```

Slurm sets `CUDA_VISIBLE_DEVICES` automatically based on your `--gres` request — you generally should **not** hardcode GPU indices yourself. On a MIG slice, `nvidia-smi` will show the specific MIG instance ID rather than a full physical GPU — that's expected and confirms you're correctly isolated to your slice.

### Single-GPU batch script template (full B200)

```bash
#!/bin/bash
#SBATCH --job-name=conformer_train
#SBATCH --partition=full-b200
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=job_%j.log

source ~/miniconda3/bin/activate patchconformer
cd $SLURM_SUBMIT_DIR

python train.py \
  --config configs/displace_patchconformer.yaml \
  --batch_size 32 \
  --num_workers $SLURM_CPUS_PER_TASK
```

### MIG-slice template (small B200, for dev/inference/notebooks)

```bash
#!/bin/bash
#SBATCH --job-name=conformer_dev
#SBATCH --partition=small-b200
#SBATCH --gres=gpu:nvidia_b200_1g.45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=job_%j.log

source ~/miniconda3/bin/activate patchconformer
cd $SLURM_SUBMIT_DIR

python quick_eval.py --config configs/displace_debug.yaml
```

Note `--num_workers $SLURM_CPUS_PER_TASK` in the first template — tying your DataLoader worker count to what you actually requested avoids CPU oversubscription, which matters more here since `--cpus-per-task` on a MIG job is a much smaller shared pool than on `full-b200`.

### On CUDA/module loading

The official cluster guide's templates leave `# Load required modules here` as a placeholder without specifying a fixed module set. Before assuming a `module load cuda/x.y` line is needed or available, run `module avail` from the login node (or inside an interactive session) to see what's actually provisioned — if nothing relevant shows up, your conda/venv-installed PyTorch build (matched to the B200's CUDA 12.8+ requirement) is likely handling CUDA itself without a separate module.

---

## 8. Multi-Node / Multi-GPU Distributed Training

For PyTorch DDP/`torchrun`-based training across multiple GPUs (or, on traditional HPC fleets, multiple nodes).

> **IIT Mandi DGX cluster note:** the current documented cluster is a single DGX0 box (4× B200, sliced into full/MIG tiers) — so the relevant scenario here is **single-node, multi-GPU** below, requesting multiple full GPUs from `full-b200`. The multi-*node* subsection further down is included for completeness and will apply if/when additional DGX nodes are added to the cluster — check with IT support before assuming it works today.

### Single node, multiple GPUs (applies now, on `full-b200`)

```bash
#!/bin/bash
#SBATCH --job-name=ddp_train
#SBATCH --partition=full-b200
#SBATCH --gres=gpu:nvidia_b200:4      # all 4 full GPUs on the DGX box
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=job_%j.log

source ~/miniconda3/bin/activate patchconformer
cd $SLURM_SUBMIT_DIR

torchrun --standalone --nproc_per_node=4 train.py --config configs/displace.yaml
```

Requesting all 4 GPUs on a single-DGX cluster means no one else can run a `full-b200` job concurrently — reserve this for runs that genuinely need multi-GPU DDP, and confirm with `dgx-status` first since you'll be waiting for full availability.

### Multi-node distributed training (forward-looking — not applicable to the current single-DGX0 cluster)

This is where the `nodes` / `ntasks` / `srun` relationship really matters — one task per node, `torchrun` handles process-spawning per GPU within each node. Keep this for reference if the cluster expands to multiple DGX nodes:

```bash
#!/bin/bash
#SBATCH --job-name=ddp_multinode
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # one launcher task per node
#SBATCH --gres=gpu:4                # 4 GPUs per node
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00

module load cuda/12.8
source ~/miniconda3/bin/activate patchconformer
cd $SLURM_SUBMIT_DIR

# Figure out the head node's address for rendezvous
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  train.py --config configs/displace.yaml
```

Key ideas:
- `srun` here launches the `torchrun` command **once per node** (because `ntasks-per-node=1`), and `torchrun` itself spawns the 4 per-GPU processes on each node.
- The rendezvous endpoint (`head_node_ip:29500`) is how nodes find each other to form the distributed process group — pick an unused port.
- `$SLURM_NNODES` and `$SLURM_JOB_NODELIST` are populated automatically by Slurm.

---

## 9. Job Arrays

Job arrays run the same script many times with different indices — perfect for hyperparameter sweeps, or running the same eval across many DISPLACE audio files.

```bash
#!/bin/bash
#SBATCH --job-name=hp_sweep
#SBATCH --partition=small-b200                  # small MIG is enough for most sweeps — frees full-b200 for real training
#SBATCH --gres=gpu:nvidia_b200_1g.45gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-9                             # 10 jobs, indices 0 through 9
#SBATCH --output=sweep_%A_%a.log                # %A = array job ID, %a = task index

source ~/miniconda3/bin/activate patchconformer
cd $SLURM_SUBMIT_DIR

LR_VALUES=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1)
LR=${LR_VALUES[$SLURM_ARRAY_TASK_ID]}

python train.py --config configs/displace.yaml --lr $LR --run_name "sweep_${SLURM_ARRAY_TASK_ID}"
```

Useful array controls:

```bash
#SBATCH --array=0-99%10     # 100 jobs, but only 10 run concurrently at once (throttling)
#SBATCH --array=1,3,5,7     # specific indices only
```

On `small-b200`, a physical B200 splits into 8 small MIG slices — so `--array=0-9%8` is a sensible throttle if you want your sweep to use as much of the tier as politely possible without starving other users entirely.

Managing an array:

```bash
squeue --me                    # shows as 123456_0, 123456_1, ...
scancel 123456                 # cancels the whole array
scancel 123456_3               # cancels just task index 3
```

---

## 10. Job Dependencies & Pipelines

Chain jobs so one only starts after another finishes — useful for preprocess → train → evaluate pipelines.

```bash
# Submit preprocessing, capture its job ID
prep_id=$(sbatch --parsable preprocess.sh)

# Train only after preprocessing succeeds
train_id=$(sbatch --parsable --dependency=afterok:$prep_id train.sh)

# Evaluate only after training succeeds
sbatch --dependency=afterok:$train_id evaluate.sh
```

Dependency types:

| Type | Runs when... |
|---|---|
| `afterok:jobid` | The specified job completed **successfully** |
| `afterany:jobid` | The specified job finished, regardless of exit code |
| `afternotok:jobid` | The specified job **failed** |
| `after:jobid` | The specified job has *started* (not necessarily finished) |
| `singleton` | No other job with the same name/user is running (useful to prevent duplicate concurrent runs) |

`--parsable` makes `sbatch` print just the bare job ID (instead of "Submitted batch job 123456"), which is what makes this chaining pattern scriptable.

---

## 11. Environments: Modules, Conda, and Virtualenvs

Most clusters use **environment modules** to expose versioned software (CUDA, compilers, MPI) without polluting the base system.

> **IIT Mandi DGX note:** the official cluster guide's job templates leave `# Load required modules here` as an open placeholder and don't document a specific module set. Run `module avail` on the login node to see what's actually provisioned before assuming a `module load cuda/x.y` line is required — many users on this cluster manage CUDA entirely through their conda/pip-installed PyTorch build instead (matched to the B200's CUDA 12.8+ requirement).

```bash
module avail                    # list everything available
module avail cuda               # search
module load cuda/12.8           # load a specific version
module list                     # what's currently loaded
module purge                    # unload everything (good practice at script start)
```

Typical pattern combining modules + conda in a batch script:

```bash
module purge
module load cuda/12.8
module load gcc/11.3

source ~/miniconda3/etc/profile.d/conda.sh
conda activate patchconformer

# sanity check before the real work starts
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Always `module purge` at the top of scripts — inherited login-shell modules are a common source of "works interactively, fails in batch" bugs.

---

## 12. Monitoring, Logs, and Debugging

### While a job runs

```bash
squeue --me                                  # is it running/pending, and why
squeue -j 123456 -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"   # custom columns
scontrol show job 123456                     # full detail: node, reason, resources
tail -f logs/train_123456.out                # live log tail while it runs
```

To watch GPU utilization live on the node your job is running on:

```bash
srun --jobid=123456 --pty nvidia-smi -l 5     # attach to a running job's node
```

(Support for attaching this way is cluster-config dependent; if it doesn't work, `ssh` directly to the node listed in `squeue` if your cluster allows direct node SSH for jobs you own.)

### After a job finishes

```bash
sacct -j 123456 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
```

Key fields to check on a mysterious failure:
- **State**: `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, `CANCELLED`, `NODE_FAIL`
- **ExitCode**: non-`0:0` means your program returned an error — check your log
- **MaxRSS**: if close to your `--mem` request, you likely OOM'd

### Common failure signatures

| Symptom | Likely cause |
|---|---|
| `State = OUT_OF_MEMORY` | `--mem` too low, or a real memory leak — check `MaxRSS` |
| `State = TIMEOUT` | `--time` too short — checkpoint and requeue, or extend |
| Job stuck `PD (Resources)` | Cluster genuinely full — wait, or request less |
| Job stuck `PD (Priority)` | Others ahead in queue — often fair-share related |
| Job stuck `PD (QOSMaxGRESPerUser)` | You've hit your personal GPU allocation limit already |
| CUDA "no kernel image" error | GPU compute capability newer than your installed PyTorch/CUDA build supports — reinstall PyTorch built for that CUDA version |
| Works with `srun --pty` but fails under `sbatch` | Environment not fully set up in the batch script (missing `module load`/`conda activate`, or a `.bashrc`-only setting not sourced in non-interactive shells) |

---

## 13. Checkpointing & Requeueing Long Jobs

For long training runs, don't fight the walltime — checkpoint and let Slurm requeue you.

### Signal-based graceful checkpoint

Request a warning signal sent before Slurm kills your job at the time limit:

```bash
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@120        # send SIGUSR1, 120 seconds before the kill
#SBATCH --requeue                  # allow this job to be requeued automatically
```

In your Python training loop, catch it and checkpoint immediately:

```python
import signal, sys

def handle_timeout(signum, frame):
    print("Received timeout warning — saving checkpoint...")
    save_checkpoint(model, optimizer, step)
    sys.exit(0)

signal.signal(signal.SIGUSR1, handle_timeout)
```

And have your script auto-resume from the latest checkpoint on (re)start:

```python
if os.path.exists(latest_checkpoint_path):
    load_checkpoint(model, optimizer, latest_checkpoint_path)
```

### Manual chained resubmission (simpler, very common)

```bash
#!/bin/bash
#SBATCH --job-name=long_train
#SBATCH --partition=full-b200
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --time=08:00:00
#SBATCH --output=job_%j.log

source ~/miniconda3/bin/activate patchconformer
cd $SLURM_SUBMIT_DIR

python train.py --resume_from_latest --config configs/displace.yaml

# If not yet converged, resubmit myself for another chunk of time
if [ ! -f "training_complete.flag" ]; then
    sbatch $0
fi
```

This pattern (a script that resubmits itself) is simple, robust, and doesn't rely on signal timing being perfectly reliable across cluster configurations.

---

## 14. Common Pitfalls

- **Confusing `--ntasks` with `--nodes`.** `--ntasks=4` on `--nodes=1` runs 4 processes on one machine; you almost never want `--ntasks` > 1 for a plain single-GPU PyTorch script.
- **Forgetting `--gres`/`--gpus` entirely** and then wondering why `torch.cuda.is_available()` is `False` — CPU-only nodes exist on most clusters, and Slurm won't give you a GPU unless you explicitly ask.
- **Requesting far more memory/time than needed "just in case."** This hurts your queue priority and availability — right-size based on `sacct` history from prior similar runs.
- **Not creating the log output directory** before `sbatch`, if you deviate from the cluster's default flat `job_%j.log` convention — job may run but produce no visible output file.
- **Leaving interactive `srun`/`salloc` sessions idle** — resources sit reserved and unused, burning fair-share priority and blocking others.
- **Hardcoding GPU device indices** (`cuda:0`) instead of trusting `CUDA_VISIBLE_DEVICES` remapping done by Slurm — breaks on shared nodes and MIG slices alike.
- **Sourcing `.bashrc` assumptions that only exist in interactive shells** — always explicitly `conda activate` (and `module load`, if applicable on this cluster) inside the batch script itself, don't rely on login-shell state.
- **Not checking `sinfo`/partition max walltime before submitting** — jobs exceeding partition limits get rejected outright at submission.
- **DGX-specific: defaulting to `full-b200` out of habit** when a MIG tier would do — this needlessly ties up a whole physical GPU and slows down everyone's queue, including your own future jobs.
- **DGX-specific: skipping `dgx-status` before submitting a `full-b200` job** — with only one DGX box, full-GPU availability is the tightest resource on this cluster; check before you queue and wait.

---

## 15. Quick Reference Cheat Sheet

```bash
# ---- Login ----
ssh <username>@dgx-login.iitmandi.ac.in

# ---- Cluster state ----
dgx-status                              # IIT Mandi: GPU/MIG availability at a glance
sinfo                                   # partitions & node states
sinfo -p full-b200 -o "%N %G"           # GPU types on the full-b200 partition
squeue --me                             # my jobs
squeue -u $USER                         # equivalent (as used in the official guide)

# ---- Submitting ----
sbatch script.sh                        # submit batch job
sbatch --parsable script.sh             # submit, print bare job ID only
sbatch --dependency=afterok:JOBID next.sh   # chained job

# ---- Interactive ----
srun --partition=small-b200 --gres=gpu:nvidia_b200_1g.45gb:1 \
     --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
salloc --partition=full-b200 --gres=gpu:nvidia_b200:1 --time=04:00:00   # then srun inside it

# ---- Managing ----
scancel JOBID                           # cancel a job
scancel -u $USER                        # cancel all my jobs
scontrol show job JOBID                 # full detail on a job
scontrol hold JOBID / scontrol release JOBID   # pause/resume a pending job

# ---- History / accounting ----
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS
sacct --starttime today -u $USER

# ---- IIT Mandi DGX partition / GRES reference ----
# full-b200    --gres=gpu:nvidia_b200:1              (whole GPU)
# medium-b200  --gres=gpu:nvidia_b200_3g.90gb:1       (90GB MIG slice)
# small-b200   --gres=gpu:nvidia_b200_1g.45gb:1       (45GB MIG slice)
# h200 tiers (full/medium/small) — coming later

# ---- Common SBATCH directives (this cluster) ----
#SBATCH --job-name=name
#SBATCH --partition=full-b200          # or medium-b200 / small-b200
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=job_%j.log
#SBATCH --array=0-9
#SBATCH --dependency=afterok:JOBID
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
```

---

## Suggested next step

Once you've read through this, the fastest way to internalize it is to open an interactive session on the DGX cluster and just poke around:

```bash
ssh <username>@dgx-login.iitmandi.ac.in
dgx-status
srun --partition=small-b200 --gres=gpu:nvidia_b200_1g.45gb:1 \
     --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
```

Run `hostname`, `nvidia-smi`, `squeue --me` from a second terminal — seeing the pieces move live is worth more than any amount of reading.

---

## Appendix: mapping to the official IIT Mandi guide

This document extends the official *IIT Mandi DGX Cluster User Guide* — for the day-to-day essentials it documents directly (login, `dgx-status`, the three job templates, `sbatch`/`squeue`/`scancel`, and support contacts for LDAP/Slurm/GPU/storage/software issues), that guide remains the authoritative source. Everything here beyond that — resource-request internals, interactive debugging patterns, distributed training, job arrays/dependencies, checkpointing, and troubleshooting — fills in the "how do I actually use this well" gaps the base guide doesn't cover. If something here ever conflicts with an update to the official guide or with what you observe on the cluster, trust the official guide and treat this as due for a refresh.
