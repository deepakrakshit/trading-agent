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

The initial `docs.pytorch.org` page appeared stale (older selector/recommendations). The current `https://pytorch.org/get-started/locally/` page and the official wheel metadata were checked instead. PyTorch **2.10.0** has official CPython **3.14** Linux wheels, so the actual host runtime **3.14.4** was kept and tested rather than replacing system Python. CPU and optional CUDA dependency sets are pinned separately. No torchvision/torchaudio, system CUDA toolkit, global pip install, or third-party driver installer was needed.

## Baseline repository state

The workspace initially contained only `self_improving_trading_agent_blueprint.md`; there was no Git repository, source tree, test suite, dependency file, notebook, model, or credential file. The blueprint was read in full (770 lines). The implementation will initialize local Git history and preserve the blueprint unchanged as the canonical specification.

## Reverification

Run these commands after any driver/runtime change:

```bash
nvidia-smi
python -m trading_agent.cli doctor
```

Do not infer GPU training from PCI detection alone.

## Completed local installation and execution evidence

| Environment | Actual PyTorch | Outcome |
| --- | --- | --- |
| `.venv` | `2.10.0+cpu` | Local install, editable package install, offline tests passed |
| `.venv-cuda` | `2.10.0+cu128`, bundled CUDA runtime 12.8 | Local install and CPU-mode tests passed; CUDA device execution failed |
| `.venv-repro` | `2.10.0+cpu` | Created empty, installed `requirements.lock` from scratch, installed project, all tests passed |

The official CUDA wheel metadata explicitly lists Python 3.14 and pins the bundled NVIDIA libraries; its source was:

`https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp314-cp314-manylinux_2_28_x86_64.whl.metadata`

Actual CUDA probe from `.venv-cuda`:

```text
torch: 2.10.0+cu128
torch.version.cuda: 12.8
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
nvidia-smi exit code: 9
CPU tensor check: 14.0
```

`.venv-cuda/bin/python -m trading_agent.cli doctor --require-cuda` exited **2**. An explicit `train --device cuda` exited **1**, before creating a training run. These are expected failed GPU checks, not skipped tests or successful GPU training.

Further diagnosis: **Secure Boot enabled**; the running kernel lacks the NVIDIA module; the installed older-kernel modules pin 595.71.05, while the current-kernel module requires 595.91.07 and the latest userspace candidate is 595.99.02. Two `apt-get --simulate install` transactions failed dependency resolution. No real apt install, root operation, driver/kernel change, Secure Boot change, or reboot was performed. See `docs/gpu_repair.md` for the exact evidence and user-controlled repair route.

## Measured compute workload

- Initial candidate: 9 observations, 5 exposure actions, **32×32 MLP (1,541 trainable parameters)**, batch 32, replay capacity 5,000, update every 4 interactions, one CPU Torch thread.
- Actual initial run: **3,000 interactions / 735 optimizer updates / 20.6236 seconds** of measured training-loop time, using **CPU**, even though launched from the CUDA-wheel venv.
- Separate 500-step benchmark: **3.7872 seconds**, CPU. Same workload can compare CPU/CUDA after driver repair; no GPU speedup or cross-device comparison is claimed now.
- Second candidate: 449 imported paper transitions plus 1,000 new interactions; 347 updates total; **6.9430 seconds** for new interaction training. Import/initialization time is not included in that number.
- Peak VRAM: **not measurable / null**. The GPU's expected 4 GiB is not a runtime-verified available-memory result.
- Remaining disk after both primary Python environments were installed: approximately **39 GiB available** at the measured checkpoint. No memory-exhausting model was used.

## Container limitation

`docker --version` returned `command not found`. The maintained Dockerfile pins Python, installs system libraries and the lock, builds the package and runs `make test`, but a Docker build/run was **not executed** on this host. Fresh-venv reproducibility was executed and verified instead; it is not presented as a container test.

## Continuation audit (20 October 2026)

- Current committed revision: `2835cd2` (`fix: harden paper timing accounting and data provenance`).
- The source archive was extracted into a clean temporary directory without
  ignored `data/`, `models/`, `runs/`, virtual-environment or cache contents.
  Running `make test PYTHON=/home/deepakrakshit/trading_agent/.venv-repro/bin/python`
  from that archive passed **249 tests**, compilation and `pip check`.
  This is a clean-source verification using an existing project-local venv,
  not a Docker build or a fresh package download.
- A fresh CPU audit candidate used `.venv` / PyTorch `2.10.0+cpu`, 64
  interactions, one optimizer update and **0.5981 seconds**. The checkpoint
  reloaded and produced valid action `4`; peak VRAM remained `null`.
- `make test PYTHON=.venv-cuda/bin/python` passed **249 tests** in the CUDA
  wheel environment, but `.venv-cuda/bin/python -m trading_agent.cli
  doctor --require-cuda` still exits 2. The driver/kernel blocker is unchanged;
  no privileged repair, reboot or GPU training was attempted.

## Completed-bar continuous paper follow-up (20 October 2026)

- Added a provider-independent `CompletedBarFeed` boundary with deterministic
  replay, finite range adaptation, explicit provider/provenance identity,
  completed-bar availability and local receipt timestamps, and bounded retries
  that never substitute synthetic data.
- Added `ContinuousPaperRunner`, which reuses `PaperBroker`,
  `PortfolioLedger`, `RiskSupervisor` and `SQLiteRepository`. Its durable
  cursor is transactionally coupled to the broker checkpoint and supports
  restart continuation, stale-feed health records, graceful stop, and a
  persistent paper-only kill switch. The CLI surface is `paper-stream`.
- The follow-up added ten tests (bringing the suite from 249 to **259
  passed**) and was verified with `.venv-repro`. A two-command synthetic
  `paper-stream`/`--resume` smoke check processed separate event batches and
  retained the same SQLite account.
- `make demo` was rerun with the new execution-v4-compatible default paths
  `models/initial-v3/` and `runs/demo-v3/`; it completed, verified restart
  recovery, and rejected the synthetic candidate at ₹9,582.25 after 260 fills
  and ₹149.70 fees. The pre-v4 v2 artifacts were preserved rather than
  overwritten.
- This is deterministic offline/replay evidence only. The live SmartAPI
  adapter remains mock-tested, no live entitlement or credentials were used,
  and no real order route was added.
