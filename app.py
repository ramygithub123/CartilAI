"""CartilAI — AI-assisted knee MRI analysis. Gradio web app.

Run with ``python app.py`` (or ``python start.py`` to install everything first).
The app runs in demo mode (untrained weights, synthetic scans) until
``models/cartilai_resnet18.pt`` exists, then switches to the trained model.
"""

from __future__ import annotations

import argparse
import html
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT
from src.data import load_volume_file
from src.gradcam import overlay
from src.inference import Analysis, Predictor, list_samples, load_metrics
from src.report import DISCLAIMER

APP_TITLE = "CartilAI"
TAGLINE = "AI-assisted knee MRI analysis."
NO_OVERLAY = "No overlay"

predictor = Predictor()
SAMPLES = list_samples()
SAMPLE_TITLES = [s["title"] for s in SAMPLES]
METRICS = load_metrics()
RESULTS_DIR = PROJECT_ROOT / predictor.config["paths"]["results_dir"]

CSS = """
:root { --ca-ink:#0f172a; --ca-muted:#475569; --ca-line:#e2e8f0; --ca-accent:#0f766e; --ca-bg:#f8fafc; }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important; }
#ca-header { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding:18px 22px; border-radius:14px; background:linear-gradient(120deg,#0b3b4a 0%,#0f766e 100%); color:#fff; }
#ca-header .brand { display:flex; align-items:center; gap:14px; }
#ca-header h1 { margin:0; font-size:30px; font-weight:700; letter-spacing:-0.5px; color:#fff; }
#ca-header h1 span { color:#99f6e4; }
#ca-header p { margin:2px 0 0; font-size:15px; color:#ccfbf1; }
.ca-badge { padding:7px 12px; border-radius:999px; font-size:13px; font-weight:600; white-space:nowrap; }
.ca-badge.demo { background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }
.ca-badge.trained { background:#dcfce7; color:#166534; border:1px solid #86efac; }
#ca-disclaimer { margin-top:10px; padding:10px 14px; border-radius:10px; background:#fff7ed; border:1px solid #fed7aa;
  color:#9a3412; font-size:14px; }
.ca-card { border:1px solid var(--ca-line); border-radius:12px; padding:14px 16px; background:#fff; color:var(--ca-ink); }
.ca-card h3 { margin:0 0 10px; font-size:15px; color:var(--ca-muted); font-weight:600; text-transform:uppercase; letter-spacing:.4px; }
.ca-row { display:grid; grid-template-columns: 130px 1fr 64px 128px; align-items:center; gap:10px; padding:7px 0; border-top:1px solid #f1f5f9; }
.ca-row:first-of-type { border-top:none; }
.ca-name { font-weight:600; font-size:15px; }
.ca-track { position:relative; height:12px; background:#f1f5f9; border-radius:6px; overflow:visible; }
.ca-fill { height:100%; border-radius:6px; }
.ca-tick { position:absolute; top:-4px; width:2px; height:20px; background:#0f172a; opacity:.55; }
.ca-pct { font-variant-numeric: tabular-nums; font-weight:700; text-align:right; }
.ca-chip { font-size:12px; font-weight:600; padding:3px 8px; border-radius:999px; text-align:center; }
.ca-chip.pos { background:#fee2e2; color:#991b1b; } .ca-chip.ind { background:#fef3c7; color:#92400e; } .ca-chip.neg { background:#dcfce7; color:#166534; }
.ca-note { font-size:12.5px; color:var(--ca-muted); margin-top:8px; }
.ca-strip { display:flex; align-items:flex-end; gap:2px; height:46px; margin-top:4px; }
.ca-strip div { flex:1; background:#99f6e4; border-radius:2px 2px 0 0; min-height:2px; }
.ca-strip div.cur { background:#0f766e; }
.ca-sample { font-size:13.5px; color:var(--body-text-color, #0f172a); opacity:.85; }
.ca-metrics table { width:100%; border-collapse:collapse; font-size:14px; }
.ca-metrics th, .ca-metrics td { border-bottom:1px solid var(--ca-line); padding:8px 10px; text-align:left; }
.ca-metrics th { color:var(--ca-muted); font-weight:600; }
/* Our panels have fixed light backgrounds, so pin their text colours; otherwise Gradio's dark theme
   recolours <b>, <i> and headings to near-white and they disappear. */
#ca-disclaimer, #ca-disclaimer * { color:#9a3412 !important; }
.ca-card, .ca-card .ca-name, .ca-card .ca-pct, .ca-card b, .ca-card strong, .ca-card i, .ca-card em { color:var(--ca-ink) !important; }
.ca-card h3, .ca-card .ca-note { color:var(--ca-muted) !important; }
.ca-card .ca-note b { color:var(--ca-ink) !important; }
.ca-chip.pos { color:#991b1b !important; } .ca-chip.ind { color:#92400e !important; } .ca-chip.neg { color:#166534 !important; }
.ca-row { border-top-color:#f1f5f9 !important; }
footer { display:none !important; }
@media (max-width: 700px) { .ca-row { grid-template-columns: 1fr 56px; } .ca-track, .ca-chip { grid-column: span 2; } }
"""

LOGO_SVG = """<svg width="44" height="44" viewBox="0 0 44 44" fill="none" aria-hidden="true">
<rect width="44" height="44" rx="11" fill="#ffffff" fill-opacity="0.12"/>
<path d="M15 6 C15 14 13 17 14 21 C15 24 19 24 21 22" stroke="#ccfbf1" stroke-width="3" stroke-linecap="round"/>
<path d="M29 38 C29 31 30 28 28 25 C26 22 22 23 21 26" stroke="#ccfbf1" stroke-width="3" stroke-linecap="round"/>
<circle cx="32" cy="12" r="3" fill="#5eead4"/><circle cx="12" cy="32" r="2" fill="#5eead4"/>
<path d="M32 12 L24 20 M12 32 L18 27" stroke="#5eead4" stroke-width="1.5"/></svg>"""


def header_html() -> str:
    """Build the page header with the name, tagline and active-mode badge."""
    if predictor.demo_mode:
        badge = '<span class="ca-badge demo">● Demo mode — untrained weights</span>'
    else:
        badge = '<span class="ca-badge trained">● Trained model loaded</span>'
    warn = ""
    if predictor.load_error:
        warn = f'<div id="ca-disclaimer">⚠ {html.escape(predictor.load_error)} — running in demo mode.</div>'
    return f"""
<div id="ca-header"><div class="brand">{LOGO_SVG}<div><h1>Cartil<span>AI</span></h1><p>{TAGLINE}</p></div></div>{badge}</div>
<div id="ca-disclaimer"><b>Research prototype — not for clinical use.</b> CartilAI estimates the probability that findings are
<i>already present</i> on an MRI exam. It does not predict future injury risk and must not be used for diagnosis.</div>{warn}"""


def findings_html(a: Analysis | None) -> str:
    """Render probability bars with threshold ticks and a call for each finding."""
    if a is None:
        return '<div class="ca-card"><h3>Findings</h3><p class="ca-note">Choose a sample scan or upload a volume, then press Analyze.</p></div>'
    rows = []
    for c in a.calls:
        chip = {"likely present": ("pos", "Likely present"), "indeterminate": ("ind", "Indeterminate"),
                "not detected": ("neg", "Not detected")}[c.category]
        color = {"pos": "#dc2626", "ind": "#d97706", "neg": "#16a34a"}[chip[0]]
        rows.append(
            f'<div class="ca-row"><div class="ca-name">{html.escape(c.name)}</div>'
            f'<div class="ca-track" title="Probability {c.probability:.1%}; decision threshold {c.threshold:.0%}">'
            f'<div class="ca-fill" style="width:{c.probability * 100:.1f}%;background:{color}"></div>'
            f'<div class="ca-tick" style="left:calc({c.threshold * 100:.1f}% - 1px)"></div></div>'
            f'<div class="ca-pct">{c.probability:.0%}</div><div class="ca-chip {chip[0]}">{chip[1]}</div></div>'
        )
    note = "Bar = model probability that the finding is present. Black tick = decision threshold"
    note += " (chosen on held-out data)." if not predictor.demo_mode else " (0.5 in demo mode)."
    if predictor.demo_mode:
        note += " <b>Demo mode: weights are untrained, so these numbers are placeholders.</b>"
    return (f'<div class="ca-card"><h3>Findings — {html.escape(a.exam_label)}</h3>{"".join(rows)}'
            f'<div class="ca-note">{note}</div></div>')


def strip_html(a: Analysis | None, finding: str, slice_no: int) -> str:
    """Mini bar chart of how much each slice contributed to the selected finding."""
    if a is None or finding == NO_OVERLAY:
        return ""
    k = [c.name for c in a.calls].index(finding)
    imp = a.importance[k]
    peak = float(imp.max()) or 1.0
    bars = "".join(
        f'<div class="{"cur" if i + 1 == slice_no else ""}" style="height:{max(4, 100 * v / peak):.0f}%" '
        f'title="Slice {i + 1}: {v:.1%} of heatmap mass"></div>' for i, v in enumerate(imp)
    )
    key = ", ".join(str(s) for s in a.calls[k].key_slices) or "none"
    return (f'<div class="ca-card"><h3>Slice relevance — {html.escape(finding)}</h3><div class="ca-strip">{bars}</div>'
            f'<div class="ca-note">Share of the Grad-CAM heat on each slice (dark = current slice). Key slices: {key}.</div></div>')


def render(a: Analysis | None, finding: str, slice_no: int, alpha: float) -> np.ndarray | None:
    """Draw the current slice with the chosen finding's heatmap."""
    if a is None:
        return None
    s = int(np.clip(slice_no, 1, a.volume.shape[0])) - 1
    cam = None
    if finding != NO_OVERLAY:
        k = [c.name for c in a.calls].index(finding)
        cam = a.cams[k, s]
    img = overlay(a.volume[s], cam, alpha=alpha, window=a.window)
    if img.shape[0] < 512:  # upscale for a crisper display
        img = np.asarray(Image.fromarray(img).resize((512, 512), Image.BILINEAR))
    return img


def best_slice(a: Analysis, finding: str) -> int:
    """1-based slice with the most Grad-CAM heat for ``finding`` (middle slice if none)."""
    if finding == NO_OVERLAY:
        return a.volume.shape[0] // 2 + 1
    k = [c.name for c in a.calls].index(finding)
    imp = a.importance[k]
    return int(np.argmax(imp)) + 1 if imp.max() > 0 else a.volume.shape[0] // 2 + 1


def default_finding(a: Analysis) -> str:
    """Show the most confident specific finding first (ACL / meniscus), else Abnormal."""
    specific = [c for c in a.calls if c.key != "abnormal"] or a.calls
    return max(specific, key=lambda c: c.probability - c.threshold).name


def _run(volume: np.ndarray, label: str, alpha: float) -> tuple[Any, ...]:
    a = predictor.analyze(volume, exam_label=label)
    finding = default_finding(a)
    s = best_slice(a, finding)
    report_path = Path(tempfile.gettempdir()) / f"cartilai_report_{abs(hash(label)) % 10**8}.txt"
    report_path.write_text(a.report, encoding="utf-8")
    return (
        a,
        findings_html(a),
        gr.Radio(value=finding),
        gr.Slider(minimum=1, maximum=a.volume.shape[0], value=s, step=1,
                  label=f"Slice (1–{a.volume.shape[0]})", interactive=True),
        render(a, finding, s, alpha),
        strip_html(a, finding, s),
        a.report,
        gr.DownloadButton(value=str(report_path), interactive=True),
    )


def sample_info(title: str) -> str:
    """Describe a bundled sample (source and reference labels)."""
    for s in SAMPLES:
        if s["title"] == title:
            kind = "Real MRNet exam" if s["kind"] == "real" else "Synthetic (cartoon) scan"
            return f'<div class="ca-sample"><b>{kind}.</b> {html.escape(s.get("description", ""))}<br>{html.escape(s.get("source", ""))}</div>'
    return ""


def analyze_sample(title: str, alpha: float) -> tuple[Any, ...]:
    """Load a bundled sample and analyse it."""
    match = [s for s in SAMPLES if s["title"] == title]
    if not match:
        raise gr.Error("Please choose a sample scan first.")
    return _run(np.load(match[0]["path"], allow_pickle=False), title, alpha)


def analyze_upload(file_path: str | None, alpha: float) -> tuple[Any, ...]:
    """Load an uploaded ``.npy`` (MRNet) or ``.pck`` (KneeMRI) volume and analyse it."""
    if not file_path:
        raise gr.Error("Upload an MRNet-format .npy volume (or a KneeMRI .pck file) first.")
    try:
        volume = load_volume_file(file_path)
        return _run(volume, Path(file_path).name, alpha)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def about_markdown() -> str:
    """Model card shown on the 'About the model' tab, filled from results/metrics.json."""
    mode = ("**Active mode: demo** — no trained weights found at `models/cartilai_resnet18.pt`, so the app uses "
            "seeded, untrained weights and synthetic scans. Add the files produced by the Kaggle notebook "
            "(see `KAGGLE_STEPS.md`) and restart to switch to the trained model."
            if predictor.demo_mode else
            f"**Active mode: trained model** — weights loaded from `{predictor.weights_path.relative_to(PROJECT_ROOT)}`.")
    parts = ["## About the model", mode, ""]
    if METRICS:
        m = METRICS
        parts += [f"### Held-out performance", f"Evaluated on: {m.get('evaluation_set', 'held-out set')}.", ""]
        parts += ["| Finding | AUC (95% CI) | Sensitivity | Specificity | Accuracy | Threshold | Positives / negatives |",
                  "|---|---|---|---|---|---|---|"]
        for f in predictor.findings:
            r = m["findings"].get(f)
            if not r:
                continue
            auc = "n/a" if r.get("auc") is None else f"{r['auc']:.3f}"
            ci = "" if r.get("auc_ci_low") is None else f" ({r['auc_ci_low']:.3f}–{r['auc_ci_high']:.3f})"
            parts.append(f"| {r.get('name', f)} | {auc}{ci} | {r['sensitivity']:.1%} | {r['specificity']:.1%} | "
                         f"{r['accuracy']:.1%} | {r['threshold']:.2f} | {r['n_positive']} / {r['n_negative']} |")
        t = m.get("training", {})
        parts += ["", f"Mean AUC: **{m.get('mean_auc', float('nan')):.3f}**. Thresholds chosen on "
                  f"{m.get('thresholds_selected_on', 'an internal validation split')}.",
                  f"Training: best epoch {t.get('best_epoch')} of {t.get('epochs_completed')} completed, "
                  f"{t.get('n_train')} training exams, {t.get('train_minutes', 0):.0f} min on {t.get('gpu') or t.get('device')}."]
        if predictor.demo_mode:
            parts += ["", "_Note: these metrics come from `results/metrics.json`, but the matching weights file is missing, "
                      "so the Analyze tab is still in demo mode._"]
    else:
        parts += ["### Held-out performance", "No results yet. Run `notebooks/cartilai_kaggle_training.ipynb` on Kaggle and "
                  "copy its `results/` folder into this project (see `KAGGLE_STEPS.md`). Metrics, ROC curves and confusion "
                  "matrices will then appear here automatically."]
    parts += [
        "", "### How it works",
        "- **Input:** one MRNet sagittal knee MRI series (`.npy`, slices × 256 × 256).",
        "- **Backbone:** ResNet18 pretrained on ImageNet (transfer learning) runs on every slice.",
        "- **Exam-level pooling:** each slice becomes a 512-number summary; the maximum across slices is kept "
        "(MRNet, Bien et al., *PLOS Medicine* 2018), so one strongly abnormal slice can drive the score.",
        "- **Outputs:** three independent probabilities (sigmoid) for *Abnormal*, *ACL tear* and *Meniscal tear*.",
        "- **Explanations:** Grad-CAM on the last convolutional block, per slice, shows which regions and slices "
        "drove each score. Heatmaps are not segmentations.",
        "- **Report:** a template turns the scores, thresholds and key slices into a draft report for a radiologist to edit.",
        "", "### Intended use and limitations",
        "- Research and education only. Not a medical device, not validated for clinical use.",
        "- Trained on a single public dataset (MRNet, 1,370 exams from one US hospital); performance on other "
        "scanners, protocols and populations is unknown.",
        "- Uses the sagittal plane only; radiologists read all planes and sequences.",
        "- Detects findings already present on the exam; it does not forecast future injury.",
        "", f"_{DISCLAIMER}_",
    ]
    return "\n".join(parts)


def build_demo() -> gr.Blocks:
    """Assemble the Gradio interface."""
    with gr.Blocks(title="CartilAI — AI-assisted knee MRI analysis") as demo:
        state = gr.State(None)
        gr.HTML(header_html())
        with gr.Tabs():
            with gr.Tab("Analyze"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=300):
                        gr.Markdown("#### 1 · Choose a scan")
                        sample = gr.Radio(SAMPLE_TITLES, value=SAMPLE_TITLES[0] if SAMPLE_TITLES else None,
                                          label="Bundled sample scans", elem_id="ca-samples")
                        info = gr.HTML(sample_info(SAMPLE_TITLES[0]) if SAMPLE_TITLES else "")
                        run_sample = gr.Button("Analyze selected sample", variant="primary", elem_id="ca-run-sample")
                        with gr.Accordion("…or upload your own scan (.npy or .pck)", open=False):
                            upload = gr.File(label="Sagittal knee MRI: MRNet .npy or KneeMRI .pck", file_types=[".npy", ".pck", ".pkl"], type="filepath")
                            run_upload = gr.Button("Analyze uploaded scan", elem_id="ca-run-upload")
                        findings = gr.HTML(findings_html(None))
                    with gr.Column(scale=6, min_width=360):
                        gr.Markdown("#### 2 · Review the slices and heatmaps")
                        heat_for = gr.Radio([predictor.names[f] for f in predictor.findings] + [NO_OVERLAY],
                                            value=predictor.names[predictor.findings[1]], label="Grad-CAM heatmap for")
                        viewer = gr.Image(label="MRI slice with Grad-CAM overlay", type="numpy", height=470,
                                          interactive=False, elem_id="ca-viewer")
                        slice_slider = gr.Slider(1, 24, value=12, step=1, label="Slice", elem_id="ca-slice")
                        with gr.Row():
                            alpha = gr.Slider(0.0, 0.9, value=0.45, step=0.05, label="Heatmap opacity", scale=3)
                            jump = gr.Button("Jump to key slice", scale=1)
                        strip = gr.HTML("")
                gr.Markdown("#### 3 · Draft report")
                with gr.Row():
                    report = gr.Textbox(label="Auto-generated draft (edit before use)", lines=20, max_lines=40,
                                        interactive=True, buttons=["copy"], elem_id="ca-report", scale=5)
                    download = gr.DownloadButton("Download report (.txt)", interactive=False, scale=1)
            with gr.Tab("About the model"):
                gr.Markdown(about_markdown(), elem_classes=["ca-metrics"])
                images = [(RESULTS_DIR / n, t) for n, t in (("roc_curves.png", "ROC curves"),
                                                             ("confusion_matrices.png", "Confusion matrices"),
                                                             ("training_curves.png", "Training curves"))]
                for path, title in images:
                    if path.exists():
                        gr.Markdown(f"#### {title}")
                        gr.Image(str(path), show_label=False, interactive=False, buttons=["download"], height=420)

        outputs = [state, findings, heat_for, slice_slider, viewer, strip, report, download]
        run_sample.click(analyze_sample, [sample, alpha], outputs)
        run_upload.click(analyze_upload, [upload, alpha], outputs)
        sample.change(sample_info, sample, info)
        demo.load(analyze_sample, [sample, alpha], outputs)

        def on_view(a: Analysis | None, f: str, s: int, al: float) -> tuple[Any, str]:
            return render(a, f, s, al), strip_html(a, f, s)

        def on_finding(a: Analysis | None, f: str, al: float) -> tuple[Any, ...]:
            if a is None:
                return gr.Slider(), None, ""
            s = best_slice(a, f)
            return gr.Slider(value=s), render(a, f, s, al), strip_html(a, f, s)

        slice_slider.change(on_view, [state, heat_for, slice_slider, alpha], [viewer, strip], show_progress="hidden")
        alpha.change(on_view, [state, heat_for, slice_slider, alpha], [viewer, strip], show_progress="hidden")
        heat_for.input(on_finding, [state, heat_for, alpha], [slice_slider, viewer, strip], show_progress="hidden")
        jump.click(on_finding, [state, heat_for, alpha], [slice_slider, viewer, strip], show_progress="hidden")
    return demo


def main() -> None:
    """Parse arguments and launch the web app."""
    parser = argparse.ArgumentParser(description="Launch the CartilAI web app")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab automatically")
    args = parser.parse_args()
    print(f"CartilAI starting in {predictor.mode_label.upper()} with {len(SAMPLES)} sample scans.")
    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate")
    build_demo().launch(server_name=args.host, server_port=args.port, share=args.share,
                        inbrowser=not args.no_browser, theme=theme, css=CSS, footer_links=[])


if __name__ == "__main__":
    main()
