# CartilAI — interview notes (plain language)

These notes explain the project without assuming any coding background, so you can talk about it confidently.

---

## 1. What the project does, in one paragraph

CartilAI is a research prototype that looks at a knee MRI and estimates how likely it is that three things are present: **any abnormality**, a **torn ACL** (the anterior cruciate ligament, a key ligament inside the knee), and a **torn meniscus** (the C-shaped cartilage cushions between the thigh bone and shin bone). For each score it shows a **heatmap** of the parts of the image that influenced the decision, lets you scroll through the MRI slice by slice, and writes a **draft report** a radiologist could edit. It is clearly labelled "not for clinical use", and it reports findings that are *already there*; it does not predict whether someone will get injured in the future.

## 2. The 30-second pitch

"Knee MRIs are common and slow to read. I built CartilAI, a prototype that flags likely ACL and meniscal tears and shows the radiologist *why*, using heatmaps and an auto-drafted report. It follows Stanford's MRNet approach: a ResNet18 network pretrained on everyday photos is fine-tuned on 1,130 knee MRIs on a free Kaggle GPU, and I evaluated it on 120 exams it never saw. The whole thing is a one-command web app, with tests, a reproducible training notebook and an honest write-up of its limitations."

## 3. How each part works

### The data: MRNet
- Stanford released **MRNet**: 1,370 knee MRI exams from Stanford Medical Center, labelled by radiologists for abnormality, ACL tear and meniscal tear.
- Each exam is a **stack of 2D images ("slices")**, like a loaf of sliced bread. We use the **sagittal** view (side view of the knee), typically 20–50 slices of 256 × 256 pixels.
- Stanford splits it into 1,130 training exams and 120 validation exams. We keep the 120 completely untouched as our **test set**, and set aside 10% of the training exams to choose settings.

### The CNN (convolutional neural network)
- A CNN is a model that learns to recognise visual patterns. Early layers detect simple things (edges, textures); deeper layers combine them into complex shapes (a ligament, a bone outline).
- We use **ResNet18**, a well-known 18-layer CNN. "Res" (residual) means it has shortcut connections that make deep networks easier to train.
- **How it reads a 3D exam:** ResNet18 looks at *each slice separately* and turns each into a summary of 512 numbers. Then, for each of those 512 features, we take the **maximum across all slices**. The idea: if a tear is clearly visible on even one slice, that strong signal survives. A final layer turns the 512 numbers into three scores, and a **sigmoid** squashes each into a probability between 0 and 1.
- The three outputs are independent, so an exam can be positive for both an ACL tear and a meniscal tear at once.

### Transfer learning
- Training a CNN from scratch needs millions of images. We have about a thousand exams.
- So we start from a ResNet18 that was already trained on **ImageNet** (1.2 million everyday photos). It already knows edges, textures and shapes; we only need to **fine-tune** it to knee anatomy. This is called **transfer learning**.
- It makes training faster, needs less data and usually gives better results. That's also what the MRNet paper did.

### Training details (if asked)
- **Loss:** binary cross-entropy, weighted so rarer findings (e.g. ACL tears) count more, because otherwise the model can "win" by mostly saying "no tear".
- **Optimiser:** Adam with a one-cycle learning-rate schedule, one exam per step (exams have different numbers of slices).
- **Augmentation:** random small rotations, shifts and flips of each exam so the model doesn't memorise exact positions.
- **Mixed precision** on the GPU for speed, and a hard **time budget** so the Kaggle run always finishes in under ~2 hours.
- **Fixed random seeds** so results are reproducible.

### Grad-CAM (the heatmaps)
- Grad-CAM answers: "Which parts of the image pushed this score up?"
- It looks at the last layer of feature maps in the CNN and measures how much each one influences a given output (using the **gradient**, i.e. how much the score would change if that feature changed). Weighting the feature maps by that influence and overlaying the result gives a heatmap.
- In CartilAI we compute it **per slice and per finding**, so the heatmap also tells you *which slices* mattered. The app shows a "slice relevance" bar and a **Jump to key slice** button.
- Important caveat: a heatmap shows influence, not a precise outline of the lesion. It's coarse (an 8 × 8 grid per slice) because the model was never told where the lesions are.

### AUC (how we measure performance)
- The model outputs a probability. To say "tear" or "no tear" you pick a cut-off (threshold). Different thresholds trade **sensitivity** (catching real tears) against **specificity** (not raising false alarms).
- The **ROC curve** plots that trade-off for every possible threshold. The **AUC** (area under the curve) summarises it in one number: the probability that a randomly chosen positive exam gets a higher score than a randomly chosen negative one. 0.5 = coin flip, 1.0 = perfect.
- We report AUC with a **95% confidence interval** from bootstrapping (re-sampling the 120 test exams 1,000 times), because 120 exams is a small test set.
- We also report sensitivity, specificity, accuracy and confusion matrices at a threshold chosen on the internal validation split (Youden's J: the point that maximises sensitivity + specificity).

### The draft report
- Built from a fixed template, not a language model, so it can't invent findings. For each finding it states the probability, the decision threshold, a category (likely present / indeterminate / not detected) and the most influential slices, then an impression and the disclaimer.

### The app
- A **Gradio** web app: pick a bundled sample (or upload a `.npy` file), click Analyze, see the probability bars, scroll slices with the heatmap overlay, and copy or download the report. The **About the model** tab reads the real metrics from the results file.
- **Demo mode vs trained mode:** if the trained weights file isn't there, the app still runs with untrained weights and synthetic scans, and says so clearly in a badge. Once the weights file is added, it switches automatically.

## 4. Why each design choice was made

| Choice | Why |
|---|---|
| MRNet approach (2D CNN per slice + max-pool) | Proven on this exact dataset; 2D pretrained networks transfer well and need far less GPU than 3D networks. |
| ResNet18 instead of the paper's AlexNet | More modern and accurate, still small enough to train on a free GPU in about an hour. |
| One model with three outputs | Shared features help all three tasks; one model is simpler to train, ship and explain. |
| Sagittal plane first | ACL and menisci are best seen on sagittal images; it keeps the first version simple. |
| Untouched test set, thresholds picked on a separate split | Prevents "grading your own homework"; the reported numbers are honest. |
| Bootstrap confidence intervals | 120 exams is small; intervals show how uncertain the numbers are. |
| Grad-CAM | Standard, model-agnostic way to show what drove a prediction; builds trust and helps spot mistakes. |
| Template report, no LLM | Every sentence traces back to a number; nothing can be hallucinated. |
| Synthetic data + demo mode | Lets the whole pipeline be tested without the real data, and lets the app be demoed anywhere. |
| Kaggle notebook generated from the source code | One source of truth: the code that's tested is the code that trains. |
| Gradio | Quick to build a clean, interactive interface in Python. |

## 5. Why it's called CartilAI

**Cartil**age + **AI**. The long-term goal is detecting cartilage damage in joints, which is hard to spot and clinically important (it's linked to osteoarthritis). This first version focuses on the findings MRNet provides labels for (abnormal, ACL, meniscus; the meniscus is itself fibrocartilage), and dedicated cartilage-lesion detection is the first item on the future-work list. Say this openly: the name reflects the direction, the current model does ACL/meniscus.

## 6. Honest limitations

- A **research prototype**, not a medical device; not validated clinically.
- Trained on **one dataset from one hospital**; it may perform worse on other scanners or populations.
- Uses **one plane** only; radiologists use all planes and sequences.
- **Small test set** (120 exams) → wide confidence intervals.
- Heatmaps are **coarse** and show influence, not exact lesion boundaries.
- "Abnormal" is a broad bucket; it doesn't name the specific problem.
- The demo's synthetic scans are cartoons; they test the software, not accuracy.
- It detects findings **present now**; it does not predict future injuries.

## 7. Ten likely interview questions, with good answers

**1. Walk me through the project.**
"CartilAI scores knee MRIs for abnormality, ACL tear and meniscal tear and explains each score with heatmaps and a draft report. I followed Stanford's MRNet approach: a pretrained ResNet18 processes each MRI slice, the strongest signal across slices is kept, and three outputs give probabilities. I trained it on MRNet in a Kaggle notebook, evaluated on 120 held-out exams with AUC and confidence intervals, and wrapped it in a Gradio app with a slice viewer. It's tested end to end and documented, including its limitations."

**2. Why use a 2D network on a 3D scan instead of a 3D CNN?**
"Pretrained 2D networks bring knowledge from millions of images, while good pretrained 3D medical models are rarer and 3D training needs much more memory and data. Treating slices as 2D images and pooling across them worked well in the MRNet paper and fits a free GPU. The trade-off is that the model doesn't explicitly use relationships between neighbouring slices; a 3D or multi-plane model is a natural next step."

**3. What is transfer learning and why did it matter here?**
"Starting from a network already trained on a large dataset, here ImageNet, and fine-tuning it on our task. With about a thousand MRI exams, training from scratch would overfit. The pretrained network already recognises edges and textures, so it learns knee anatomy faster and generalises better."

**4. Why max-pooling across slices?**
"A tear might only be visible on a few slices. Averaging would dilute that signal with all the normal slices; taking the maximum keeps it. The downside is sensitivity to a single noisy slice, which is one reason heatmaps and human review matter."

**5. How did you evaluate it, and how do you know you didn't overfit?**
"I split MRNet's training set 90/10 and used the 10% to pick the best epoch and decision thresholds. The official 120-exam validation set was used only once at the end, as a test set. I report AUC with bootstrap 95% confidence intervals plus sensitivity, specificity and confusion matrices, and I compare against the MRNet paper's published numbers."

**6. What does AUC mean, and why not just accuracy?**
"AUC is the probability that a random positive case gets a higher score than a random negative case; it measures ranking quality across all thresholds. Accuracy depends on one threshold and can be misleading with imbalanced classes: if only 20% of exams have an ACL tear, a model that always says 'no tear' is 80% accurate and useless. So I lead with AUC and show sensitivity and specificity at a chosen threshold."

**7. How does Grad-CAM work, and can you trust the heatmaps?**
"It takes the gradient of a finding's score with respect to the last convolutional feature maps, uses those to weight the maps, and overlays the result. It's useful for sanity checks: if the model highlights the ACL region for an ACL tear, that's reassuring; if it highlights a label in the corner, that's a red flag. But it's coarse and shows influence, not causation or exact boundaries, so I present it as an aid, not proof."

**8. How did you handle class imbalance?**
"Tears are less common than normal findings, so I weighted the loss for each finding by the ratio of negatives to positives. I also chose thresholds with Youden's J on a validation split rather than using a default 0.5, and I report sensitivity and specificity separately."

**9. What would it take to use this in a hospital?**
"A lot. External validation on data from other hospitals and scanners, subgroup analysis for bias, calibration, a prospective reader study showing radiologists do better with it, regulatory clearance (for example FDA 510(k) or CE marking), DICOM and PACS integration, monitoring for drift, and clear governance on who is accountable. It would always be decision support, with the radiologist making the call."

**10. What would you do next?**
"Four things: add the coronal and axial planes and fuse them, as the MRNet paper did; add dedicated cartilage-lesion detection, which is the project's namesake; try segmentation models to outline structures precisely; and extend the pipeline to other joints like shoulder and ankle. I'd also validate on an external dataset."

**Bonus — "What was the hardest part?"**
"Making it reproducible and honest for someone else to run: a single Kaggle notebook that runs with no edits and always finishes in time, a demo mode that works before any training, and an evaluation that doesn't leak the test set into model selection. I generated the notebook from the same tested source code so they can't drift apart."

**Bonus — "Is this predicting knee injuries?"**
"No, and I was deliberate about that. It estimates whether a finding is already visible on the scan, with a confidence score. Predicting future injury would need different data, such as follow-up outcomes, and a different study design."
