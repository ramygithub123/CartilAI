# CartilAI — 2-minute live demo script

**Before you start:** run `python start.py`, wait for the browser tab at <http://127.0.0.1:7860>, and click **Analyze selected sample** once so everything is warm. If the trained model is installed, the header badge reads **Trained model loaded** and real MRNet exams are at the top of the sample list; use those. Otherwise you're in demo mode; say so (line in brackets below).

---

### 0:00 – 0:15 · The problem
> "Knee MRIs are among the most common MRI exams, and each one has hundreds of images. CartilAI is a research prototype I built that flags likely ACL and meniscal tears and shows the radiologist *why*."

*Point at the header:* the CartilAI name, the tagline, and the red-orange **Research prototype — not for clinical use** banner.

> "It's explicitly a research tool. It estimates whether a finding is already present on the scan; it doesn't predict future injury."

### 0:15 – 0:40 · Run an exam
*Select an exam with an ACL tear (e.g. "MRNet exam … Reference labels: Abnormal, ACL tear", or "Synthetic knee — ACL tear + effusion"), click **Analyze selected sample**.*

> "I pick a knee MRI and click Analyze. In a couple of seconds I get a probability for each finding: abnormal, ACL tear, meniscal tear. The black tick on each bar is the decision threshold, chosen on held-out data, and the label says likely present, indeterminate or not detected."

*[Demo mode only:]* > "This laptop is running in demo mode with untrained weights, so the numbers are placeholders; the trained model's results are on the About tab."

### 0:40 – 1:10 · Heatmaps and slice viewer
*Make sure **ACL tear** is selected under "Grad-CAM heatmap for". Drag the **Slice** slider across a few slices, then click **Jump to key slice**.*

> "This is the MRI, one slice at a time, from one side of the knee to the other. The colour overlay is Grad-CAM: the regions that pushed the ACL score up. The bar chart underneath shows which slices mattered most, and 'Jump to key slice' takes me straight there. Here it's focusing around the intercondylar notch, where the ACL runs."

*Switch the heatmap to **Meniscal tear**, and drag **Heatmap opacity** down and up once.*

> "Each finding has its own heatmap, and I can fade it to see the anatomy underneath. These maps show influence, not exact lesion outlines, so they're an aid for the reader, not a segmentation."

### 1:10 – 1:30 · Draft report
*Scroll to **3 · Draft report**.*

> "CartilAI drafts a structured report: each finding with its probability, threshold and key slices, an impression, and the disclaimer. It's template-based, with no language model, so it can't invent findings. The radiologist edits it here, then copies or downloads it."

### 1:30 – 1:55 · How it works and how well
*Click the **About the model** tab.*

> "Under the hood it follows Stanford's MRNet paper: a ResNet18 pretrained on ImageNet looks at every slice, the strongest signal across slices is kept, and three outputs give the probabilities. I trained it on the MRNet dataset in a Kaggle GPU notebook and tested it on 120 exams it never saw." *Point at the table and ROC curves.* "Here are the AUCs with 95% confidence intervals, plus sensitivity and specificity."

### 1:55 – 2:00 · Close
> "Next steps are multi-plane fusion, dedicated cartilage-lesion detection, which is where the name comes from, and validation on data from other hospitals. Happy to go deeper on any part."

---

**If something goes wrong:** refresh the page (it re-analyses the first sample automatically). If the app isn't running, run `python start.py` again. Keep a screenshot of the About tab as a backup.
