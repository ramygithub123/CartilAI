"""Draft radiology-style report generated from model outputs (template-based, no LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np

DISCLAIMER = (
    "Research prototype — not for clinical use. This draft was generated automatically by "
    "CartilAI and has not been reviewed by a radiologist. It describes the model's confidence "
    "that findings are present on this exam; it does not predict future injury risk."
)

FINDING_PHRASES = {
    "abnormal": ("an abnormality of the knee", "No abnormality detected by the model"),
    "acl": ("a tear of the anterior cruciate ligament (ACL)", "No ACL tear detected by the model"),
    "meniscus": ("a meniscal tear", "No meniscal tear detected by the model"),
}


@dataclass
class FindingCall:
    """The model's call for one finding."""

    key: str
    name: str
    probability: float
    threshold: float
    category: str          # "likely present" | "indeterminate" | "not detected"
    key_slices: list[int]  # 1-based slice numbers that drove the prediction


def categorize(prob: float, threshold: float, margin: float) -> str:
    """Bucket a probability relative to its decision threshold.

    Args:
        prob: Model probability.
        threshold: Decision threshold for the finding.
        margin: Half-width of the "indeterminate" band around the threshold.

    Returns:
        ``"likely present"``, ``"indeterminate"`` or ``"not detected"``.
    """
    if prob >= min(0.99, threshold + margin):
        return "likely present"
    if prob <= max(0.01, threshold - margin):
        return "not detected"
    return "indeterminate"


def top_slices(importance: np.ndarray, k: int = 3) -> list[int]:
    """Return the 1-based indices of the ``k`` most important slices, in anatomical order."""
    if importance.size == 0 or float(importance.max()) <= 0:
        return []
    idx = np.argsort(importance)[::-1][:k]
    return sorted(int(i) + 1 for i in idx if importance[i] > 0)


def make_calls(
    probs: Sequence[float],
    thresholds: Sequence[float],
    findings: Sequence[str],
    names: dict[str, str],
    importance: np.ndarray,
    margin: float = 0.15,
) -> list[FindingCall]:
    """Turn raw probabilities into per-finding calls.

    Args:
        probs: Probability per finding.
        thresholds: Decision threshold per finding.
        findings: Finding keys.
        names: Display names.
        importance: ``(k, S)`` slice importance from Grad-CAM.
        margin: Indeterminate band half-width.

    Returns:
        One :class:`FindingCall` per finding.
    """
    return [
        FindingCall(f, names.get(f, f), float(p), float(t), categorize(float(p), float(t), margin), top_slices(importance[i]))
        for i, (f, p, t) in enumerate(zip(findings, probs, thresholds))
    ]


def _lower(name: str) -> str:
    """Lower-case a finding name but keep acronyms such as ACL."""
    return " ".join(w if w.isupper() and len(w) > 1 else w.lower() for w in name.split())


def _slices_text(slices: list[int]) -> str:
    if not slices:
        return ""
    return f" Most influential slices: {', '.join(str(s) for s in slices)}."


def generate_report(
    calls: list[FindingCall],
    n_slices: int,
    plane: str = "sagittal",
    exam_label: str = "Uploaded exam",
    demo_mode: bool = False,
    model_label: str = "CartilAI ResNet18 (MRNet-style)",
    now: datetime | None = None,
) -> str:
    """Write a structured draft report in plain text.

    Args:
        calls: Output of :func:`make_calls`.
        n_slices: Number of slices in the volume.
        plane: Imaging plane.
        exam_label: Free-text exam identifier.
        demo_mode: Whether untrained demo weights produced the outputs.
        model_label: Model description.
        now: Timestamp (defaults to the current time).

    Returns:
        The report text.
    """
    now = now or datetime.now()
    lines = [
        "CARTILAI — DRAFT KNEE MRI REPORT (AI-GENERATED)",
        "=" * 48,
        f"Exam:        {exam_label}",
        f"Sequence:    {plane.capitalize()} series, {n_slices} slices",
        f"Model:       {model_label}" + ("  [DEMO MODE: untrained weights]" if demo_mode else ""),
        f"Generated:   {now.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    if demo_mode:
        lines += [
            "*** DEMO MODE: the model has not been trained yet. Probabilities below are",
            "*** placeholders that only demonstrate the workflow; they carry no meaning.",
            "",
        ]
    lines.append("FINDINGS")
    by_key = {c.key: c for c in calls}
    for c in calls:
        positive, negative = FINDING_PHRASES.get(c.key, (_lower(c.name), f"No {_lower(c.name)} detected by the model"))
        pct, thr = f"{c.probability:.0%}", f"{c.threshold:.0%}"
        if c.category == "likely present":
            text = f"Model output is suggestive of {positive} (probability {pct}; decision threshold {thr})."
            text += _slices_text(c.key_slices)
        elif c.category == "indeterminate":
            text = (f"Indeterminate for {positive} (probability {pct}, close to the decision threshold of {thr}). "
                    "Recommend careful review.") + _slices_text(c.key_slices)
        else:
            text = f"{negative} (probability {pct}; decision threshold {thr})."
        lines.append(f"- {c.name}: {text}")
    lines += ["", "IMPRESSION"]
    present = [c for c in calls if c.category == "likely present" and c.key != "abnormal"]
    unsure = [c for c in calls if c.category == "indeterminate"]
    n = 1
    for c in present:
        lines.append(f"{n}. Findings suggestive of {FINDING_PHRASES.get(c.key, (_lower(c.name),))[0]} "
                     f"({c.probability:.0%} model confidence).")
        n += 1
    ab = by_key.get("abnormal")
    if ab and ab.category == "likely present" and not present:
        lines.append(f"{n}. Abnormal exam without a specific ACL or meniscal tear identified by the model "
                     f"({ab.probability:.0%} confidence). Other pathology (e.g. effusion, bone marrow edema, "
                     "cartilage lesion) may be present.")
        n += 1
    for c in unsure:
        lines.append(f"{n}. Indeterminate for {FINDING_PHRASES.get(c.key, (_lower(c.name),))[0]}; "
                     "correlate clinically and on other sequences.")
        n += 1
    if n == 1:
        lines.append("1. No ACL tear, meniscal tear or other abnormality detected by the model.")
    if ab and ab.category == "not detected" and present:
        lines.append("Note: specific tear flagged while the global abnormality score is low; outputs are "
                     "inconsistent and need review.")
    lines += [
        "",
        "Heatmaps (Grad-CAM) show image regions that most influenced each score; they are not segmentations.",
        "",
        "DISCLAIMER: " + DISCLAIMER,
    ]
    return "\n".join(lines)
