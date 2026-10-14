# Training CartilAI on Kaggle — step by step

This guide assumes no coding. You will upload one notebook file, attach the dataset, switch on a free GPU, run it, and download one zip file. Total hands-on time is about 10 minutes; the run itself takes roughly 1 to 1.5 hours.

**Dataset used:** **MRNet v1** by *cjinny* — <https://www.kaggle.com/datasets/cjinny/mrnet-v1> (Kaggle id `cjinny/mrnet-v1`). It is the widely used Kaggle copy of Stanford's MRNet-v1.0 knee MRI dataset. The notebook finds the data automatically, so any other MRNet copy with the standard layout (`train/`, `valid/`, `train-acl.csv`, …) also works.

**File you need from this project:** `notebooks/cartilai_kaggle_training.ipynb`

---

## Part 1 — One-time account setup

1. Go to <https://www.kaggle.com> and click **Register** (or **Sign In**). Signing in with Google is fine.
2. **Verify your phone number.** Kaggle only allows GPUs and Internet access in notebooks after phone verification.
   - Click your **profile picture** (top right) → **Settings**.
   - Scroll to **Phone verification** and click **Verify**. Enter your number and the code you receive by SMS.
   - If you skip this, the GPU and Internet options in Part 3 will be greyed out.

## Part 2 — Create the notebook from the file

1. On the Kaggle home page, click **+ Create** (left sidebar) → **New Notebook**. An editor opens with a sample notebook.
2. In the editor's top menu click **File** → **Import Notebook**.
3. Click **Browse Files** (or drag the file in) and choose `notebooks/cartilai_kaggle_training.ipynb` from the CartilAI project folder. Click **Import**.
4. The sample cells are replaced by the CartilAI notebook. Its first cell starts with "CartilAI — Kaggle training notebook".
5. Optional: click the notebook title at the top left and rename it to `CartilAI training`.

## Part 3 — Attach the dataset, GPU and Internet

All of this is in the **right-hand panel** of the editor. If you don't see it, click the small **|<** arrow at the far right edge of the screen to open it.

1. **Attach the dataset**
   - In the right panel find **Input** and click **+ Add Input**.
   - In the search box type `mrnet-v1`.
   - Find **MRNet v1** by **cjinny** and click the **+** (plus) button next to it.
   - It now appears under **Input** as `mrnet-v1`. Close the search box.
2. **Turn on the GPU**
   - In the right panel open **Session options** (on some layouts this is under **Settings**).
   - Under **Accelerator**, choose **GPU T4 x2**. Confirm if asked.
   - Use **T4**, not P100: the P100 is not supported by recent PyTorch versions. (If you pick it by mistake the notebook stops at the first cell with a message telling you to switch.)
3. **Turn on Internet**
   - In the same panel, switch **Internet** to **On**. This lets the notebook download the ImageNet-pretrained ResNet18 weights (about 45 MB).
   - If the toggle is greyed out, complete the phone verification in Part 1.

## Part 4 — Run everything

Choose **one** of these two ways. Option A is safer because it keeps running even if you close the browser.

**Option A (recommended): Save & Run All in the background**
1. Click **Save Version** (top right).
2. Select **Save & Run All (Commit)** and click **Save**.
3. A small box at the bottom left shows the run is in progress. You can close the tab; the run continues for up to 12 hours (this one needs about 1–1.5 hours).
4. When it finishes you'll get a notification, and the version shows as complete in **Your Work** → your notebook.

**Option B: Run All in the editor**
1. Click **Run All** in the toolbar (the ▶▶ button, or **Run** → **Run All**).
2. Keep the browser tab open and your computer awake until the last cell prints `DONE.` Interactive sessions can stop if left idle for too long.

What you'll see while it runs: the first cell prints `GPU: Tesla T4`; training prints one line per epoch with the internal validation AUC; at the end a results table and three charts (ROC curves, confusion matrices, training curves) appear. Training stops automatically before 85 minutes even if not all epochs are done.

## Part 5 — Download the results

**If you used Option A (Save Version):**
1. Open the notebook from **Your Work** (or click the notification).
2. Click the **Output** tab (or scroll down to the **Output** section).
3. Find **`cartilai_outputs.zip`** and click its **download** icon (⤓).

**If you used Option B (Run All in the editor):**
1. In the right panel open **Output** (`/kaggle/working`). Click the refresh icon if it looks empty.
2. Hover over **`cartilai_outputs.zip`**, click the **⋮** menu → **Download**.

You only need `cartilai_outputs.zip`. It contains:

```
cartilai_outputs/
├── models/cartilai_resnet18.pt        trained weights + decision thresholds (~45 MB)
├── results/metrics.json               AUCs, sensitivity, specificity, thresholds, training log
├── results/roc_curves.png
├── results/confusion_matrices.png
├── results/training_curves.png
├── results/predictions.csv            per-exam probabilities on the held-out set
└── samples/mrnet/                     6 real held-out MRNet scans + samples.json for the demo
```

## Part 6 — Put the files in the CartilAI project

1. Move **`cartilai_outputs.zip`** (don't unzip it) into the **CartilAI project folder**: the top-level folder that contains `app.py`, `start.py` and `README.md`.
2. Run the launch command in that folder:

   ```bash
   python start.py
   ```

   It detects the zip, copies everything into `models/`, `results/` and `samples/mrnet/`, fills in the results table in `README.md`, renames the zip to `cartilai_outputs.installed.zip`, and starts the app.
3. The badge in the app header changes from **Demo mode — untrained weights** to **Trained model loaded**, the six real MRNet scans appear at the top of the sample list, and the **About the model** tab shows the metrics and charts.

**Manual alternative:** unzip the file and copy the three folders from inside `cartilai_outputs/` into the project folder, merging with the existing ones:
`cartilai_outputs/models/` → `models/`, `cartilai_outputs/results/` → `results/`, `cartilai_outputs/samples/mrnet/` → `samples/mrnet/`. Then restart the app.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| GPU or Internet options are greyed out | Verify your phone number (Part 1), then reload the notebook. |
| First cell says "No GPU found" | Session options → Accelerator → **GPU T4 x2**, then Run All again. |
| First cell says the GPU "does not work with this PyTorch version" | You picked P100. Switch to **GPU T4 x2** and run again. |
| "Could not find an MRNet dataset" | The dataset isn't attached. Repeat Part 3, step 1. |
| "Could not load ImageNet weights" in the log | Internet is off. The run still works but results will be worse; turn Internet on and run again. |
| No GPU quota left | Kaggle gives about 30 GPU hours per week; the quota resets weekly (see your profile → Settings). |
| `cartilai_outputs.zip` is missing | The run didn't finish. Open the version's **Logs** to see the last message. |

MRNet is released by Stanford for non-commercial research under a research use agreement. Keep the exported scans and the trained weights private (they are excluded from git by `.gitignore`).
