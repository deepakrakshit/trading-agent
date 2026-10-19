# RTX 2050: actual driver blocker and safe recovery

## Evidence (19 October 2026)

- PCI detects NVIDIA GA107 / GeForce RTX 2050.
- Running kernel: `7.0.0-34-generic`.
- `modinfo -k 7.0.0-34-generic nvidia` fails: module not found.
- Installed module for `7.0.0-28-generic` is **595.71.05**.
- `nvidia-smi` cannot communicate with the driver.
- **Secure Boot is enabled**. Do not disable it as a shortcut.
- Ubuntu's package index offers `linux-modules-nvidia-595-open-7.0.0-34-generic` version `7.0.0-34.34+1`, but it requires NVIDIA kernel-common **595.91.07**, whereas installed older kernel modules pin **595.71.05**. The latest userspace candidate is a third version, **595.99.02**.

Two non-privileged installation simulations failed with incompatible dependency assignments. No package was changed, removed, downgraded, or force-installed:

```bash
apt-get --simulate install linux-modules-nvidia-595-open-7.0.0-34-generic nvidia-driver-595-open
apt-get --simulate install linux-modules-nvidia-595-open-7.0.0-34-generic nvidia-driver-595-open=595.91.07-0ubuntu0.26.04.1 nvidia-kernel-common-595=595.91.07-0ubuntu0.26.04.1
```

## User-controlled system repair

This session may install dependencies **inside the project only**; privileged driver/kernel modifications and rebooting require your local action. Do not run a guessed `sudo apt install` command against this conflicting package state.

1. Save your work. Open Ubuntu's **Software Updater** and **Software & Updates → Additional Drivers**.
2. Refresh/update to a consistent Ubuntu-supported kernel and recommended signed NVIDIA driver/module set. Review any removals before confirming; do not use NVIDIA's `.run` installer, random PPAs, `--force`, or disable Secure Boot.
3. Reboot into that matching kernel. If offered a Machine Owner Key enrollment screen, follow Ubuntu's official enrollment instructions locally.
4. Verify `nvidia-smi` succeeds. The current known alternate is kernel `7.0.0-28`, which already has a matching 595.71.05 module, but it should not replace installing current security updates permanently.

If the graphical updater also reports dependency conflicts, a local administrator must resolve the installed older NVIDIA module packages and matching supported kernel as one coherent package transaction. The evidence above identifies the conflicting versions precisely; it is not safe to force an automated partial upgrade.

## Project CUDA environment

`requirements-cuda.lock` pins PyTorch **2.10.0+cu128** and the CUDA libraries declared by its official wheel metadata. It supports CPython 3.14 and Ampere GPUs; no full system CUDA toolkit or `nvcc` is necessary for these binary wheels.

```bash
python3 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install --cache-dir .cache/pip -r requirements-cuda.lock
.venv-cuda/bin/python -m pip install --no-deps --no-build-isolation -e .
make test PYTHON=.venv-cuda/bin/python
.venv-cuda/bin/python -m trading_agent.cli doctor --require-cuda
```

The `doctor --require-cuda` command must run a real tensor operation on the GPU and exit nonzero if it cannot. Installing the CUDA wheel alone does not repair the missing kernel module and is not GPU-training evidence.

Keep `.venv` as a CPU fallback. After driver repair, use `--device cuda` explicitly or the CLI's verified `auto` selection; training must fail rather than silently run on CPU when `cuda` was explicitly requested.
