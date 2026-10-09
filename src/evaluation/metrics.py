"""Binary detector metrics; fake=1 is the positive class, scores are fake probabilities."""
import numpy as np


def _arrays(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or labels.shape != scores.shape or set(labels.tolist()) != {0, 1}
            or not np.isfinite(scores).all()):
        raise ValueError('Need finite scores and both real (0) and fake (1) labels')
    return labels, scores


def roc_auc(labels, scores):
    """Probability that a random fake scores above a random real; ties count half."""
    labels, scores = _arrays(labels, scores)
    fake, real = scores[labels == 1][:, None], scores[labels == 0][None, :]
    return float(((fake > real).sum() + .5 * (fake == real).sum()) / (fake.size * real.size))


def equal_error_rate(labels, scores):
    """(EER, threshold) where the real false-alarm rate and the fake miss rate meet."""
    labels, scores = _arrays(labels, scores)
    best = None
    for threshold in np.unique(scores):
        false_alarm = np.mean(scores[labels == 0] >= threshold)
        miss = np.mean(scores[labels == 1] < threshold)
        if best is None or abs(false_alarm - miss) < best[0]:
            best = (abs(false_alarm - miss), float((false_alarm + miss) / 2), float(threshold))
    return best[1], best[2]


def threshold_report(labels, scores, threshold):
    """Fixed-threshold rates; choose the threshold on validation, never on test."""
    labels, scores = _arrays(labels, scores)
    fake = scores >= threshold
    false_alarm, miss = float(fake[labels == 0].mean()), float((~fake[labels == 1]).mean())
    return dict(accuracy=float((fake == labels).mean()), balanced_accuracy=1 - (false_alarm + miss) / 2,
                real_false_alarm=false_alarm, fake_miss=miss)
