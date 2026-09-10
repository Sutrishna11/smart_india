"""
Evaluate SlopeSense's demo model on a held-out synthetic test set.

This does NOT measure real-world landslide-prediction accuracy — the model
is trained and tested on synthetic data drawn from the same made-up
distribution (see the "Real-data upgrade" section of README.md for how to
replace it with real labelled observations). What this script does confirm
is that the training loop actually learns something and that precision/
recall/F1/ROC-AUC are computed correctly, using only the standard library.

Run:  python evaluate_model.py
"""

import random

from ml_model import LogisticRegression, sigmoid


def make_dataset(n, seed):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        rain24 = rng.uniform(5, 260)
        rain72 = min(420, rain24 + rng.uniform(10, 220))
        intensity = rng.uniform(0.5, 55)
        moisture = rng.uniform(15, 98)
        slope = rng.uniform(8, 58)
        elevation = rng.uniform(100, 2200)
        drainage = rng.uniform(0, 1)
        history = rng.uniform(0, 1)
        tilt = rng.uniform(0, 6)
        forecast = rng.uniform(0, 180)
        latent = (
            0.012 * rain24 + 0.006 * rain72 + 0.030 * intensity +
            0.025 * moisture + 0.055 * slope + 0.012 * tilt +
            0.012 * forecast + 1.3 * history - 2.0 * drainage - 4.5
        )
        p = sigmoid(latent)
        label = 1 if rng.random() < p else 0
        X.append([rain24, rain72, intensity, moisture, slope, elevation,
                  drainage, history, tilt, forecast])
        y.append(label)
    return X, y


def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true)
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def roc_auc(y_true, y_prob):
    """Rank-based AUC (equivalent to the Mann-Whitney U statistic) — no external deps."""
    pos = [p for t, p in zip(y_true, y_prob) if t == 1]
    neg = [p for t, p in zip(y_true, y_prob) if t == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def main():
    X_train, y_train = make_dataset(2600, seed=42)   # same seed/size as ml_model.py's training
    X_test, y_test = make_dataset(700, seed=99)       # disjoint seed -> held-out set

    model = LogisticRegression().fit(X_train, y_train)
    y_prob = [model.predict_proba(row) for row in X_test]

    m = classification_metrics(y_test, y_prob)
    auc = roc_auc(y_test, y_prob)

    print(f"SlopeSense demo model — held-out synthetic test set (n={len(y_test)})")
    print("-" * 58)
    print(f"Accuracy : {m['accuracy']:.3f}")
    print(f"Precision: {m['precision']:.3f}")
    print(f"Recall   : {m['recall']:.3f}")
    print(f"F1 score : {m['f1']:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print(f"Confusion matrix — TP:{m['tp']}  FP:{m['fp']}  FN:{m['fn']}  TN:{m['tn']}")
    print()
    print("Reminder: this set is generated from the same synthetic distribution")
    print("used for training, so these numbers only confirm the model can learn")
    print("its own latent function. They say nothing about real landslide")
    print("prediction accuracy — replace with real labelled data before trusting")
    print("this operationally.")


if __name__ == "__main__":
    main()
