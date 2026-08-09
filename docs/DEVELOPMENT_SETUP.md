# CNN Accelerator — Development Setup

This document describes the reproducible development environment for the
CNN Accelerator project targeting the AMD Kria KV260 / XCK26 platform.

## 1. Project Baseline

### Target hardware

- Board: AMD Kria KV260
- FPGA: XCK26
- Required device: `xck26-sfvc784-2LV-c`

### HDL and FPGA toolchain

- HDL: SystemVerilog
- FPGA toolchain: AMD/Xilinx Vivado ML 2023.1
- Simulation tools:
  - `xvlog`
  - `xelab`
  - `xsim`

### Host operating system

- Ubuntu 22.04 LTS 64-bit
- x86_64

### Python / reference-model environment

The Python environment is CPU-only and is used for model/reference development,
golden-model generation, and related offline work. It is not a CUDA environment.

Required versions:

- Python: 3.10.12
- PyTorch: 2.1.2+cpu
- torchvision: 0.16.2+cpu
- NumPy: 1.26.4

Do not upgrade these versions unless compatibility testing demonstrates that a
change is required.

## 2. Vivado ML 2023.1 Setup

Install Vivado ML 2023.1 and ensure the KV260/XCK26 device database is available.

Required device: `xck26-sfvc784-2LV-c`

Expected installation location on the reference machine:
`~/Xilinx/Vivado/2023.1`

Before using Vivado or its simulator:
```bash
source ~/Xilinx/Vivado/2023.1/settings64.sh
```

Verify the installation:
```bash
vivado -version
xvlog -version
xelab -version
xsim -version
```

All four tools should report Vivado 2023.1.

Do not switch to a newer Vivado release without explicit project approval.

## 3. Python Environment

Create the project virtual environment at `~/cnn-accelerator/.venv` and activate it before running Python development commands.

Required versions:

- Python: 3.10.12
- pip: 26.2.1 or compatible current pip
- PyTorch: 2.1.2+cpu
- torchvision: 0.16.2+cpu
- NumPy: 1.26.4

The project uses a CPU-only Python environment. CUDA is not required for the development host.

Do not blindly upgrade the ML stack. Preserve the required versions unless compatibility testing demonstrates that a change is necessary.

## 4. Development Tools

The development environment requires:

- Git
- GitHub CLI (`gh`)
- GCC / G++
- GNU Make
- Tcl
- Perl
- curl
- wget
- rsync
- zip / unzip
- `uv` for installing and managing the Graphify tool
- Claude Code

These tools should be available on the system PATH.

## 5. Graphify

Graphify is used to maintain the project knowledge graph for Claude Code.

The reference environment uses:

- Graphify: 0.8.39
- Installed through `uv` as the `graphifyy` tool package

Verify the installation:

```bash
graphify --version
uv tool list
```

The repository contains the project Graphify configuration in `CLAUDE.md` and `.claude/`.
Generated Graphify data is stored in `graphify-out/` and must not be committed.

## 6. Claude Code

Claude Code is the project AI development assistant.

The reference environment uses Claude Code 2.1.207.

Project-specific Claude configuration is stored in:

- `CLAUDE.md`
- `.claude/settings.json`
- `.claude/skills/`

Verify Claude Code:

```bash
claude --version
claude plugin list
```

The project configuration enables the project-scoped `code-review` and `skill-creator` plugins.
User-scoped plugins are not part of the project setup and should not be treated as project dependencies.

## 7. First-Time Project Setup

After cloning the repository:

```bash
git clone https://github.com/ShashankT-ECE/cnn-accelerator.git
cd cnn-accelerator
```

Create and activate the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages using the versions specified in Section 3.

After cloning, the repository-provided `CLAUDE.md` and `.claude/` configuration are available automatically.
Project-scoped Claude configuration is therefore shared through Git.

## 6.1 Claude Code Project Plugins

The project uses these project-scoped plugins:

- `code-review@claude-plugins-official`
- `skill-creator@claude-plugins-official`

After cloning, open the repository in Claude Code and trust the project when prompted. If Claude Code asks you to install or approve a project plugin, accept it at project/repository scope.

After plugin installation or approval, run `/reload-plugins`.

The Research skill is included under `.claude/skills/`.

User-scoped plugins are not project dependencies.

## 5.1 Graphify Teammate Setup

Install the same Graphify version used by the project:

```bash
uv tool install graphifyy==0.8.39
```

Verify:

```bash
graphify --version
```

Expected:

```text
graphify 0.8.39
```

The repository already contains the Graphify Claude Code configuration. Do not run `graphify claude install` after cloning unless the project configuration has intentionally been removed or regenerated.

Generate the local knowledge graph when starting work:

```bash
graphify update .
```

The generated `graphify-out/` directory is local build data and is ignored by Git.

## 8. Git Workflow

Before starting work:

```bash
git pull origin main
git status
```

Create/use a feature branch for development. Do not work directly on `main` unless explicitly agreed by the team.

Recommended branch names:

- `feature/convolution`
- `feature/pooling`
- `feature/testbench`
- `feature/python-model`
- `feature/quantization`
- `feature/vivado-build`

Before pushing changes, run the relevant simulation/tests, commit using your own Git identity, push the feature branch, and open a pull request.

## 9. Hardware Development Workflow

For RTL work, follow this order:

1. Understand the architecture requirements.
2. Identify the module/interface being implemented.
3. Define interfaces and parameters.
4. Write synthesizable SystemVerilog.
5. Create or update the appropriate testbench.
6. Run Vivado simulation with `xvlog`, `xelab`, and `xsim`.
7. Check functional correctness.
8. Proceed to synthesis/implementation only after simulation is correct.
9. Optimize only after correctness is established.

Keep generated Vivado files out of Git.

## 10. Environment Verification

A new developer should be able to verify the core environment with:

```bash
# Python
source .venv/bin/activate
python --version
python -c "import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"

# Vivado
source ~/Xilinx/Vivado/2023.1/settings64.sh
vivado -version
xvlog -version
xelab -version
xsim -version

# Graphify
graphify --version

# Git
 git status
```

If these checks pass, the basic development environment is ready.
