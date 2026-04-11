# Team Setup Instructions

One-time setup to replicate the repo and verify the OpenScene environment. **Estimated time: 30–45 min on a GPU machine.**

## Before you start

- ✅ Accept your GitHub repo invite (check your email for an invite from `a1bhinav/repopt-3d`). You can't push until you accept.
- ✅ You have a machine with a GPU and CUDA installed.
- ✅ You have conda or Miniconda installed.
- ✅ Your laptop has an SSH key registered with GitHub. Test with `ssh -T git@github.com` — you should see "Hi <username>! You've successfully authenticated...". If it fails, see the FAQ at the bottom.

## Step 1 — Clone the repo with submodules

```bash
git clone --recurse-submodules git@github.com:a1bhinav/repopt-3d.git
cd repopt-3d
```

If you forgot `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

## Step 2 — Create the conda environment

```bash
conda create -n repopt python=3.11 -y
conda activate repopt
```

Check your CUDA version with `nvidia-smi` (look at "CUDA Version" in the top-right of the output). Then install PyTorch matching it.

For **CUDA 12.x** (most newer GPUs):
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
```

For **CUDA 11.8**:
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
```

## Step 3 — Install OpenScene's dependencies

```bash
pip install --no-cache-dir scipy open3d ftfy tensorboardx tqdm imageio plyfile opencv-python sharedarray git+https://github.com/openai/CLIP.git
```

This takes ~3 minutes. `open3d` is the biggest download.

**If you hit a `blinker` distutils error at the end of the install**, run this once and then re-run the install above:
```bash
pip install --no-cache-dir --ignore-installed blinker
```

## Step 4 — Run the verification command

```bash
python -c "import torch, scipy, open3d, ftfy, tensorboardX, tqdm, imageio, plyfile, cv2, SharedArray, clip; print('All OpenScene deps OK | torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

Expected output:
All OpenScene deps OK | torch: 2.4.1+cu124 | CUDA: True

**If you see `CUDA: False`**, your torch install is CPU-only — your CUDA version probably didn't match. Re-do Step 2 with the right `cu118`/`cu124` selection.

**If you see `ModuleNotFoundError: No module named 'X'`**, install just that one package: `pip install X` and re-run the verification.

**If anything else goes wrong**, post in the team channel with the exact error message.

## Step 5 — Create your branch

```bash
git checkout -b setup/<your-firstname-lowercase>
```

For example: `git checkout -b setup/soorya`

## Step 6 — Update the checkpoint file

Open `team/setup-checkpoint.md` and **fill in only your own row** in the status table. Use Abhinav's row as an example.

Columns to fill:
- **Verification output** — the `All OpenScene deps OK` part of the command output (you don't need to include the torch/CUDA part, those have their own columns)
- **Torch version** — e.g. `2.4.1+cu124`
- **CUDA available** — `True` or `False`
- **Date** — today, in YYYY-MM-DD format
- **Notes** — your GPU model, OS, and any issues you hit + how you fixed them. Be specific. Future-you will thank you.

## Step 7 — Commit and push your branch

```bash
git add team/setup-checkpoint.md
git commit -m "Setup checkpoint: <your name>"
git push -u origin setup/<your-firstname-lowercase>
```

## Step 8 — Open a pull request

Go to the repo on GitHub. You'll see a yellow banner offering to open a PR from your branch. Click it.
- **Title:** `Setup checkpoint: <your name>`
- **Description:** anything notable about your setup, or just leave blank
- Click "Create pull request"

Abhinav will review and merge.

## Convention reminder

- **Don't push directly to `main`.** Always go through a PR. This is a team norm, not technically enforced — please respect it.
- **One PR per task.** Don't bundle unrelated changes.
- **Branch naming:** `setup/<firstname>` for this task. We'll formalize naming for real work later.

## Deadline

**Monday EOD.** If you finish early, great. If you're stuck, ping the team channel **before Sunday night** so there's time to help.

## FAQ

**`Permission denied (publickey)` when cloning**
Your laptop's SSH key isn't on GitHub. Generate one:
```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```
Press Enter through all the prompts (default location, no passphrase). Then:
```bash
cat ~/.ssh/id_ed25519.pub
```
Copy the output and add it at https://github.com/settings/keys → New SSH key. Then retry the clone.

**`nvcc not found` during torch install**
Ignore it. You don't need `nvcc` for the install — only the runtime CUDA libraries, which `nvidia-smi` confirms you already have.

**`conda: command not found`**
Install Miniconda from https://docs.conda.io/en/latest/miniconda.html. Then close and reopen your terminal.

**Repo invite expired**
Ping Abhinav, he'll resend.

**My GPU is too old / I don't have CUDA 11.8 or 12.x**
Post in the team channel with your `nvidia-smi` output. We'll figure out a path forward.
