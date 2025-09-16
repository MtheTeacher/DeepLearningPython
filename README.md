# Deep Learning Python – Classroom Edition

This repository modernizes the exercises from [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/chap1.html) so the material can be demonstrated live in the classroom.  The legacy NumPy/Theano scripts are still available for reference, but the default experience now uses PyTorch together with an interactive terminal dashboard that visualizes gradient descent in real time.

## Highlights

- **PyTorch training stack** – CUDA-ready models for RTX 20/30/40/50-series GPUs with optional mixed precision.
- **Live Rich dashboard** – animated loss and accuracy charts, gradient norms, step-by-step logs, and ASCII art previews of the current mini-batch so students can “see” what the network is learning.
- **Checkpointing & metrics export** – periodic checkpoints capture the model, optimizer, and scheduler states; JSONL metrics logs can be replayed or plotted later.
- **Typer CLI** – launch scripted runs or quick classroom demos with a single command.

## Getting started

1. **Install Python** – Python 3.10 or newer is recommended.
2. **Set up a virtual environment** (optional but encouraged):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**.  Choose the right PyTorch build for your hardware.  The example below installs the CPU-only wheel; replace the `pip install torch ...` command with the [CUDA-specific instructions from pytorch.org](https://pytorch.org/get-started/locally/) if you have an NVIDIA GPU.

   ```bash
   pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -e .
   ```

   Installing the project in editable mode exposes the `dlp` command line tool described below.

## Live training demo

The new command line interface lives at `src/deeplearning_python/cli.py`.  After installing the package you can launch the interactive trainer:

```bash
dlp train --epochs 5 --batch-size 128 --learning-rate 0.05 --model simple
```

The trainer will:

- stream loss/accuracy plots, gradient norms, and learning rate updates using Rich’s live layout
- display ASCII renderings of the digits in the current mini-batch together with predicted labels
- write checkpoints to `./artifacts/checkpoints/step_XXXXXXX.pt`
- append structured metrics to `./artifacts/logs/metrics.jsonl`

### Demo mode

Need a quick classroom walkthrough?  `dlp demo` caps the run to a small number of steps, increases logging frequency, and still produces visual updates:

```bash
dlp demo --steps 100 --batch-size 64
```

### Customizing the experiment

Common options (see `dlp train --help` for the full list):

- `--model`: `simple`, `regularized`, or `conv`
- `--hidden-sizes`: adjust the layer sizes of the simple MLP (e.g. `--hidden-sizes 256 128 64`)
- `--scheduler`: enable `steplr` or `cosine` learning rate schedules
- `--checkpoint-interval`: how many optimizer steps to wait between checkpoints
- `--preview-interval`: how often to refresh the mini-batch ASCII preview
- `--mixed-precision`: turn on CUDA AMP for faster GPU demos
- `--fake-data`: run against lightweight synthetic data when you do not have network access

All options are compatible with CPU-only environments, so instructors can rehearse on a laptop before moving to the classroom GPU workstation.

## Checkpoints and metrics

Checkpoints capture the model, optimizer, scheduler, and metadata and are saved in the directory passed with `--checkpoint-dir`.  Resume training later with:

```bash
dlp train --resume-from artifacts/checkpoints/step_0000200.pt
```

The JSONL metrics file contains structured entries for both training and validation.  You can load it in pandas, Excel, or your visualization tool of choice for post-class analysis.

## Legacy scripts

The original `network.py`, `network2.py`, `network3.py`, and `test.py` files remain untouched for historical reference.  They rely on Python 3.5 and Theano and are no longer maintained, but feel free to keep them for comparison during lessons.

## Development

Install the optional development extras and run the test suite:

```bash
pip install -e .[dev]
pytest
```

Pull requests and classroom-inspired enhancements are welcome!
