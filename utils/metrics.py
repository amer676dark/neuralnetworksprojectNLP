"""Evaluation metrics for Arabic ASR: WER, CER, and more."""

from typing import List, Dict
import jiwer
import numpy as np


# Arabic text normalization transforms for WER
WER_TRANSFORMS = jiwer.Compose([
    jiwer.RemovePunctuation(),
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute Word Error Rate (WER).
    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=total words in reference.
    """
    return jiwer.wer(
        references,
        hypotheses,
        reference_transform=WER_TRANSFORMS,
        hypothesis_transform=WER_TRANSFORMS,
    )


def compute_cer(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute Character Error Rate (CER).
    More suitable for Arabic as word boundaries can be ambiguous.
    """
    return jiwer.cer(references, hypotheses)


def compute_detailed_wer(reference: str, hypothesis: str) -> Dict:
    """Return detailed WER breakdown: substitutions, deletions, insertions."""
    measures = jiwer.compute_measures(reference, hypothesis)
    return {
        "wer": measures["wer"],
        "substitutions": measures["substitutions"],
        "deletions": measures["deletions"],
        "insertions": measures["insertions"],
        "hits": measures["hits"],
        "reference_length": measures["substitutions"] + measures["deletions"] + measures["hits"],
    }


def compute_batch_metrics(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, float]:
    """Compute all metrics for a batch of predictions."""
    wer = compute_wer(references, hypotheses)
    cer = compute_cer(references, hypotheses)

    # Per-sample WER
    per_sample_wer = [
        compute_wer([ref], [hyp])
        for ref, hyp in zip(references, hypotheses)
    ]

    return {
        "wer": wer,
        "cer": cer,
        "mean_sample_wer": float(np.mean(per_sample_wer)),
        "std_sample_wer": float(np.std(per_sample_wer)),
        "min_wer": float(np.min(per_sample_wer)),
        "max_wer": float(np.max(per_sample_wer)),
    }


def format_metrics_report(metrics: Dict) -> str:
    """Format metrics dict into a readable report string."""
    lines = ["=" * 50, "Evaluation Results", "=" * 50]
    for key, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"  {key:<25}: {val:.4f} ({val*100:.2f}%)")
        else:
            lines.append(f"  {key:<25}: {val}")
    lines.append("=" * 50)
    return "\n".join(lines)
