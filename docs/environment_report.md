# Environment report

Generated during the initial repository audit on **2026-10-19 16:48 IST**. Values below are observations from this laptop, not assumptions.

## Operating system and hardware

| Check | Observed value |
| --- | --- |
| Kernel | `Linux Acer-Aspire-7 7.0.0-34-generic #34-Ubuntu SMP PREEMPT_DYNAMIC Wed Sep 2 14:29:37 UTC 2026 x86_64` |
| Distribution | Ubuntu 26.04 LTS (Resolute Raccoon) |
| CPU | 13th Gen Intel(R) Core(TM) i5-13420H |
| Logical CPUs | 12 |
| Physical cores reported | 8 |
| RAM | 14 GiB total; 11 GiB available at audit time |
| Swap | 4 GiB total; unused at audit time |
| Workspace filesystem | 73 GiB total, 49 GiB available (29% used) |
| Python | 3.14.4 at `/usr/bin/python3` |
| pip | 25.1.1, system executable; project installs use `.venv` only |
| Git | 2.53.0 |
| Node/npm | Not installed (`node` and `npm` commands unavailable) |
| `venv` module | Available |

## GPU/CUDA evidence

The PCI bus reports an NVIDIA device:

```text
01:00.0 VGA compatible controller: NVIDIA Corporation GA107 [GeForce RTX 2050] (rev a1)
```

However, CUDA is **not verified or usable in the current session**:

```text
$ nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
Make sure that the latest NVIDIA driver is installed and running.

$ nvcc --version
nvcc: command not found

$ modinfo nvidia
modinfo: ERROR: Module nvidia not found.
```

The installed package inventory contains NVIDIA 595.71.05 userspace/driver packages and modules for kernels `7.0.0-27` and `7.0.0-28`, while the running kernel is `7.0.0-34-generic`. No NVIDIA module was found under the current kernel's module directory. This is a driver/kernel packaging issue, not proof that the GPU is unavailable at the hardware level.

At audit time PyTorch was not installed, so `torch.cuda.is_available()` could not yet be called. After the project environment is installed, the verification command is:

```bash
python -m trading_agent.cli doctor
```

The project will keep CPU as the tested fallback. A CUDA result will only be reported if both `nvidia-smi` and the PyTorch CUDA probe succeed. Repairing or switching the kernel/driver may require a user-controlled reboot or privileged package operation and is intentionally not attempted automatically.

## Dependency decision

The official PyTorch installation page (`https://docs.pytorch.org/get-started/locally/`) recommends Python 3.9–3.12 for Linux. This host has Python 3.14.4 and no alternative Python binary. The project therefore tests the newest compatible wheels in an isolated `.venv`, records the exact resolved versions in `requirements.lock`, and keeps a Dockerfile pinned to the same declared application runtime. If a future wheel drops Python 3.14 support, use the documented Docker path or install a supported 3.12 interpreter locally; do not alter system Python.

## Baseline repository state

The workspace initially contained only `self_improving_trading_agent_blueprint.md`; there was no Git repository, source tree, test suite, dependency file, notebook, model, or credential file. The blueprint was read in full (770 lines). The implementation will initialize local Git history and preserve the blueprint unchanged as the canonical specification.

## Reverification

Run these commands after any driver/runtime change:

```bash
nvidia-smi
python -m trading_agent.cli doctor
```

Do not infer GPU training from PCI detection alone.
