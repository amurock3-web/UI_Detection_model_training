# Train the model on Google Colab — step by step

Takes about 2–3 hours, most of it waiting. Do the steps in order.

---

## A. Before you start (only once)

1. On your computer, find `data/renamed.zip` in this repo folder (about 131 MB).
2. Open **drive.google.com**.
3. Click **New → New folder**, name it exactly `ui16_data`, click **Create**.
4. Open the `ui16_data` folder.
5. Drag `renamed.zip` into it. Wait until the upload finishes.

You should now have: **My Drive → ui16_data → renamed.zip**

> The zip contains the original labels. The 35 label fixes come from GitHub and
> are applied automatically in cell 5. You do not need to re-zip anything.

---

## B. Open the notebook

1. Go to **colab.research.google.com**.
2. In the window that opens, click the **GitHub** tab.
   (No window? Menu **File → Open notebook → GitHub**.)
3. In the search box type `amurock3-web` and press Enter.
4. Pick repository **amurock3-web/UI_Detection_model_training**, branch **master**.
5. Click **notebooks/train_colab.ipynb**.

---

## C. Turn on the GPU

1. Menu **Runtime → Change runtime type**.
2. Under **Hardware accelerator**, pick **T4 GPU**.
3. Click **Save**.

---

## D. Run the cells, one at a time

Click the **▶** button on the left of each code cell. Wait for it to finish
(the ▶ stops spinning) and check the output before you run the next one.

| Cell | What it does | You should see |
|---|---|---|
| **1. GPU check** | Checks the GPU | `GPU    : Tesla T4` and `VRAM   : 15.6 GB` |
| **2. Clone** | Gets the code from GitHub | `cwd: /content/UI_Detection_model_training` and `sha: ...` |
| **3. Dependencies** | Installs libraries (1–2 min) | `Setup complete ✅` and `torch CUDA still available: True` |
| **4. Dataset** | Loads your zip from Drive | A popup asks to connect Drive → click **Connect to Google Drive**, pick your account, click **Allow**. Then: `OK: 103 images, 103 labels, classes_22.txt present` |
| **5. Remap + split** | Applies label fixes, splits 83/20 | `Split matches the local run exactly: 20 val images` |
| **6. runs/ → Drive** | Saves training output to Drive as it goes | `runs/ -> /content/drive/MyDrive/ui16_runs` |
| **7. TRAIN** | Trains, then zips and downloads | Lines scroll for 1.5–2.5 hours (see E) |

**Do not run cell 8.** It's for a later experiment.

---

## E. While it trains

1. Keep the Colab tab open. Leave the computer awake (don't let it sleep).
2. Check back now and then. If Colab shows **"Are you still there?"**, click
   **Yes**/**Connect**.
3. What the lines mean:
   - `Epoch 37/200` — 200 is the maximum. It usually stops sooner.
   - `mAP50` / `mAP50-95` — the score. Higher is better.
   - `EarlyStopping: Training stopped early` — normal. The model stopped
     getting better for 50 epochs.
4. When it's done you'll see `train.py exit code: 0` and `elapsed: ... min`.

---

## F. After it finishes

1. Your browser downloads several files. If it asks, click **Allow** (multiple downloads).
   - `ui16_runs_<date>.zip` — every run saved in Drive
   - one `best_ui16_<run date>_<stamp>.pt` for **each** run in Drive (the old
     one too). The new model is the one whose run date is today.
2. Unzip `ui16_runs_<date>.zip` into this repo's `runs/` folder.
   You should end up with `runs/detect/ui16_<date>_<time>/weights/best.pt`.
   (That zip holds every run in `MyDrive/ui16_runs`, including the old one. The new one is the folder with the latest date.)
3. In a terminal, from the repo folder, run:

   ```bash
   python scripts/evaluate.py runs/detect/ui16_<date>_<time>
   python scripts/error_analysis.py runs/detect/ui16_<date>_<time>
   ```

4. Tell Claude the folder name. Next we compare it with the first run.

---

## G. If something goes wrong

| You see | Do this |
|---|---|
| `NO GPU` (cell 1) | Do step C again, then **Runtime → Run all** |
| `Not found: /content/drive/MyDrive/ui16_data/renamed.zip` (cell 4) | Check the Drive folder is named exactly `ui16_data` and the file is exactly `renamed.zip`. Re-run cell 4 |
| `DATASET NOT USABLE` (cell 4) | The zip is wrong or incomplete. Upload `data/renamed.zip` again |
| `SPLIT DRIFTED` (cell 5) | Stop. Don't train. Tell Claude the full message |
| `CUDA out of memory` (cell 7) | Tell Claude. The fix is `batch: 4` in `config/train.yaml` |
| Session disconnected mid-training | Nothing is lost up to the last improvement: open Drive → `ui16_runs` → `detect` → newest folder → `weights/best.pt`. Download that folder |
| Downloads didn't start | Click the **folder icon** on Colab's left side, open `/content`, right-click `ui16_runs_<date>.zip` → **Download** |
| `pip` replaced torch (cell 3) | Menu **Runtime → Restart session**, then run cells 1–7 again |
