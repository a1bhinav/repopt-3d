# Team Setup Checkpoint

Each team member fills in their row after replicating the repo and creating their conda environment. This is a one-time setup verification — see the task message Abhinav sent for instructions.

## Status table

| Name     | Verification output | Torch version | CUDA available | Date | Notes |
|----------|--------------------|--------------:|:--------------:|------|-------|
| Abhinav  |                    |               |                |      |       |
| Soorya   |                    |               |                |      |       |
| Khalit   |                    |               |                |      |       |
| Chinmai  |                    |               |                |      |       |
| Tanvi    |                    |               |                |      |       |
| Yusuf    |                    |               |                |      |       |
| Seelan   |                    |               |                |      |       |

## How to fill your row

1. Follow the setup instructions in Abhinav's task message.
2. Run the verification command in your activated conda env.
3. Edit only your own row in the table above:
   - **Verification output:** the exact text the command printed
   - **Torch version:** e.g. `2.4.1+cu124`
   - **CUDA available:** `True` or `False`
   - **Date:** YYYY-MM-DD
   - **Notes:** GPU model, OS, conda env name if not `repopt`, any issues you hit and how you fixed them
4. Commit on your branch and open a PR.

## Convention

- `main` is for reviewed, working code only. Don't push to main directly — open a PR instead. (This isn't technically enforced yet, just a team norm. Please respect it.)
- Branch naming: `setup/<your-firstname-lowercase>` for this task. We'll figure out more conventions later.
- One PR per task. Don't bundle unrelated changes.
