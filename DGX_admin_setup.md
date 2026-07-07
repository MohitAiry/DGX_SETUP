# IIT Mandi DGX Cluster: Admin Setup Guide for B200 PyTorch Support

**Audience:** Cluster system administrators and technicians  
**Purpose:** This document outlines three robust, production-grade solutions that will permanently resolve the PyTorch/CUDA compatibility issues encountered by users on the NVIDIA B200 (Blackwell/`sm_100`) nodes. Implementing any one of these options will eliminate the need for users to manually handle wheel files, nightly builds, or CUDA version conflicts.

---

## Why This is Needed

The NVIDIA B200 GPUs use the **Blackwell architecture (`sm_100`)**, which requires **CUDA 12.8+** compiled PyTorch binaries. As of early 2026, only **PyTorch Nightly builds** ship with this support. The stable PyTorch 2.6 release (which is the one users install by default) contains no `sm_100` kernel image and causes a fatal runtime error the moment a user tries to use the GPU.

Currently, every user must individually:
1. Hunt down the correct daily nightly wheel URLs
2. Download massive 2.5GB+ files over the login node's unstable internet connection
3. Fight `pip`'s dependency resolver to prevent it from silently downgrading their installation

The three options below resolve this at the system level so it is **done once by an admin and never again by any user**.

---

## Option 1 (Recommended): Apptainer/Singularity Containers via NVIDIA NGC

### What is this?
**Apptainer** (formerly Singularity) is the industry-standard container runtime for HPC clusters. Unlike Docker (which requires root access and is insecure on multi-user systems), Apptainer containers can be run as a regular user. **NVIDIA GPU Cloud (NGC)** maintains a library of perfectly pre-built Docker images that are heavily optimized for every specific GPU architecture, including the latest Blackwell hardware.

The idea is simple: the admin downloads the correct NVIDIA PyTorch container image **once**, converts it to an Apptainer `.sif` (Singularity Image Format) file, and places it in a shared, read-only directory. Users then run their code **inside** this container. All CUDA libraries, PyTorch, and drivers are baked in — there is nothing to install.

### Prerequisites
- Apptainer installed on all nodes (check with `apptainer --version`)
- Internet access from the head node, or the `.sif` file transferred manually
- A shared filesystem accessible from all compute nodes (e.g., `/shared/`, `/opt/`, or NFS)

### Step-by-Step Admin Instructions

**Step 1: Find the correct NGC PyTorch image**

Navigate to [https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) and select the most recent tag that explicitly lists **CUDA 12.8** and **Python 3.10+** support. The tag format is typically `YY.MM-py3`.

As a reference, a valid tag looks like:
```
nvcr.io/nvidia/pytorch:25.02-py3
```

**Step 2: Pull and convert the Docker image to an Apptainer `.sif`**

Run the following from the **head/login node** (this requires internet access). This conversion can take 10–20 minutes:
```bash
# Recommended: place the final image on the shared filesystem
SHARED_DIR="/opt/shared/containers"
mkdir -p $SHARED_DIR

# Pull from NGC and build the SIF file directly
apptainer build $SHARED_DIR/pytorch_b200_cu128.sif docker://nvcr.io/nvidia/pytorch:25.02-py3
```

> **Note:** The resulting `.sif` file may be 15–30 GB in size. Ensure you have sufficient space on the shared filesystem.

**Step 3: Verify B200 support inside the container**

Before announcing availability, verify that the `sm_100` architecture is correctly supported:
```bash
# Launch an interactive shell inside the container (--nv passes through the GPU)
apptainer shell --nv $SHARED_DIR/pytorch_b200_cu128.sif

# Inside the container, run this check:
python -c "import torch; print(torch.cuda.get_arch_list()); print(torch.version.cuda)"
```
Confirm that `sm_100` appears in the output list and that the CUDA version is `12.8` or higher.

**Step 4: Set permissions**
```bash
chmod 755 $SHARED_DIR/pytorch_b200_cu128.sif
chown root:users $SHARED_DIR/pytorch_b200_cu128.sif
```

### How Users Would Use This
Users simply add the `apptainer exec` wrapper to their SLURM scripts. No installation needed at all:
```bash
#!/bin/bash
#SBATCH --partition=small-b200
#SBATCH --gres=gpu:nvidia_b200_1g.45gb:1

apptainer exec --nv /opt/shared/containers/pytorch_b200_cu128.sif \
    python /home/$USER/Projects/MyProject/train.py
```
The `--nv` flag binds the host system's NVIDIA GPU drivers directly into the container, giving it full GPU access.

### Maintenance
- When a new stable PyTorch release officially supports `sm_100`, simply repeat Steps 1–3 to generate a new `.sif` file with a new name. Old images remain functional and untouched.

---

## Option 2: Global Pre-Built Shared Conda Environment

### What is this?
Instead of each user downloading gigabytes of libraries independently, the admin creates a single Conda environment in a globally-readable directory with the correct PyTorch Nightly and CUDA 12.8 dependencies. Users then **clone** this base environment into their personal space instantly, without needing any internet access.

### Prerequisites
- A working `conda` or `mamba` installation on the login/head node
- A shared filesystem accessible from all nodes (e.g., `/opt/shared/`)
- Sufficient disk space (~5–8 GB for the base environment)

### Step-by-Step Admin Instructions

**Step 1: Create the base Conda environment**
```bash
conda create -n pytorch-b200-base python=3.10 -y
conda activate pytorch-b200-base
```

**Step 2: Download the correct Nightly wheels with resumable `wget`**

The `wget -c` flag makes these resumable if the connection drops:
```bash
# Create a temporary download directory
mkdir -p /tmp/torch_wheels && cd /tmp/torch_wheels

wget -c https://download-r2.pytorch.org/whl/nightly/cu128/torch-2.12.0.dev20260407%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
wget -c https://download-r2.pytorch.org/whl/nightly/cu128/torchvision-0.27.0.dev20260407%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
wget -c https://download-r2.pytorch.org/whl/nightly/cu128/torchaudio-2.11.0.dev20260407%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl
wget -c https://files.pythonhosted.org/packages/3b/52/94aecda69df65ba1079a8b7dbe84632af5614dc0ed2c733185f6431874e3/nvidia_cudnn_cu12-9.20.0.48-py3-none-manylinux_2_27_x86_64.whl
```

**Step 3: Install PyTorch into the base environment**
```bash
conda activate pytorch-b200-base

# Install cuDNN first as a local wheel
pip install /tmp/torch_wheels/nvidia_cudnn_cu12-9.20.0.48-py3-none-manylinux_2_27_x86_64.whl

# Install the three matching PyTorch packages together
pip install \
    /tmp/torch_wheels/torch-2.12.0.dev20260407+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
    /tmp/torch_wheels/torchvision-0.27.0.dev20260407+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
    /tmp/torch_wheels/torchaudio-2.11.0.dev20260407+cu128-cp310-cp310-manylinux_2_28_x86_64.whl \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

**Step 4: Verify the installation**
```bash
python -c "import torch; print('sm_100 supported:', 'sm_100' in torch.cuda.get_arch_list()); print('CUDA:', torch.version.cuda)"
```
The output must confirm `sm_100 supported: True`.

**Step 5: Move the environment to the shared location and lock it**
```bash
SHARED_ENV_DIR="/opt/shared/conda_envs"
mkdir -p $SHARED_ENV_DIR

# Copy the environment to the shared location
cp -r ~/miniconda3/envs/pytorch-b200-base $SHARED_ENV_DIR/

# Make it read-only to prevent accidental modifications
chmod -R a-w $SHARED_ENV_DIR/pytorch-b200-base
```

**Step 6: Announce the path to users**

Users can clone the shared environment into their own personal space with a single command:
```bash
conda create --name my_project_env --clone /opt/shared/conda_envs/pytorch-b200-base
```
This clone operation is entirely **local** (no internet needed) and completes in seconds.

### Maintenance
When a new, stable PyTorch version with `sm_100` support is released (expected in PyTorch 2.14 or 3.0 stable), repeat Steps 1–5 to create a new versioned environment (e.g., `pytorch-b200-v2`) alongside the old one.

---

## Option 3: Environment Modules via Lmod

### What is this?
**Lmod** (Lua-based Module system) is the standard software management system used by virtually every major HPC facility in the world (including TACC, NERSC, and Supercomputing centers). It allows admins to install multiple software stacks and lets users load them on-demand with `module load <package>` — a single command that instantly configures all environment variables, library paths, and executable paths.

### Prerequisites
- Lmod installed on the cluster (check with `module --version`)
- A shared filesystem (e.g., `/opt/modulefiles/`, `/share/modules/`)
- Root access on the cluster

### Step-by-Step Admin Instructions

**Step 1: Install the PyTorch software stack**

Follow the same procedure as Option 2 Steps 1–4 to install the correct PyTorch Nightly into a directory like `/opt/software/pytorch/2.12-cuda12.8/`. The key difference here is that the **environment directory itself is the "module."**

```bash
# Install the conda environment at a versioned path
conda create --prefix /opt/software/pytorch/2.12-cuda12.8 python=3.10 -y
conda activate /opt/software/pytorch/2.12-cuda12.8

# ... (follow the same pip install steps from Option 2)
```

**Step 2: Create the Lmod modulefile**

Create the directory structure for the module files:
```bash
mkdir -p /opt/modulefiles/pytorch
```

Create the modulefile at `/opt/modulefiles/pytorch/2.12-cuda12.8.lua`:
```lua
-- /opt/modulefiles/pytorch/2.12-cuda12.8.lua
-- Lmod modulefile for PyTorch 2.12 Nightly with CUDA 12.8 (B200/sm_100 support)

help([[
  PyTorch 2.12.0 Nightly with CUDA 12.8 support.
  Fully compatible with NVIDIA B200 (Blackwell, sm_100) GPUs.
  
  Usage:
    module load pytorch/2.12-cuda12.8
]])

whatis("Name:        PyTorch")
whatis("Version:     2.12.0.dev20260407+cu128")
whatis("CUDA:        12.8")
whatis("GPU Arch:    sm_100 (Blackwell B200)")

-- Define the software installation path
local base = "/opt/software/pytorch/2.12-cuda12.8"

-- Set environment variables that activate the Conda environment
prepend_path("PATH",            pathJoin(base, "bin"))
prepend_path("LD_LIBRARY_PATH", pathJoin(base, "lib"))
prepend_path("LIBRARY_PATH",    pathJoin(base, "lib"))
prepend_path("PYTHONPATH",      pathJoin(base, "lib/python3.10/site-packages"))
setenv("CONDA_PREFIX",          base)
setenv("CUDA_HOME",             "/usr/local/cuda-12.8")
setenv("TORCH_CUDA_ARCH_LIST",  "9.0;10.0")  -- Hopper + Blackwell
```

**Step 3: Register the module path with Lmod**

Add the module path to the system-wide Lmod configuration so all users can discover it:
```bash
# Add this line to /etc/profile.d/z00_lmod.sh or equivalent
echo 'export MODULEPATH=/opt/modulefiles:$MODULEPATH' >> /etc/profile.d/z00_lmod.sh
```

Or if using Lmod's `LMOD_MODULEPATH_DEFAULT`, add it directly:
```bash
# In /etc/lmod/lmodrc.lua
prepend_path("MODULEPATH", "/opt/modulefiles")
```

**Step 4: Verify the module is discoverable**
```bash
module spider pytorch
```
You should see `pytorch/2.12-cuda12.8` listed.

### How Users Would Use This
This is the simplest possible experience for any user. They just add one line to their SLURM script:
```bash
#!/bin/bash
#SBATCH --partition=small-b200
#SBATCH --gres=gpu:nvidia_b200_1g.45gb:1

module load pytorch/2.12-cuda12.8

python /home/$USER/Projects/MyProject/train.py
```
That's it. No Conda activation, no wheel files, no internet access required.

### Maintenance
When a new version is needed, create a new versioned modulefile (e.g., `pytorch/2.14-cuda12.8.lua`) pointing to a new installation. Users can switch versions trivially with `module switch pytorch/2.12-cuda12.8 pytorch/2.14-cuda12.8`.

---

## Summary & Recommendation

| Feature | Option 1: Apptainer | Option 2: Shared Conda | Option 3: Lmod |
|---|---|---|---|
| **User experience** | ⭐⭐⭐⭐⭐ Zero setup | ⭐⭐⭐⭐ One-line clone | ⭐⭐⭐⭐⭐ `module load` |
| **Admin effort** | Medium (one-time) | Low (one-time) | Medium (one-time) |
| **Reproducibility** | Excellent (image is frozen) | Good | Good |
| **Isolation** | Perfect (container) | Good (per-user clone) | Shared state |
| **Disk space** | ~25 GB per image | ~5 GB base env | ~5 GB |
| **Requires root** | For installation only | No | Yes |

**Recommendation:** Implement **Option 1 (Apptainer)** as the primary solution for maximal reproducibility and true environment isolation. Complement it with **Option 3 (Lmod)** for users who prefer a native Conda workflow. Option 2 is a good quick-win if Apptainer and Lmod are not yet installed on the cluster.
