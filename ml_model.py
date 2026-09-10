"""Lightweight, dependency-free ML model for SlopeSense.

This demo uses logistic regression trained at startup on synthetic historical
examples. Replace train_demo_model() with real labelled landslide observations
for a production deployment.
"""
import math
import random

FEATURES = [
    "rain24", "rain72", "intensity", "moisture", "slope_angle",
    "elevation", "drainage", "history", "tilt_rate", "forecast6h"
]


def sigmoid(z):
    z = max(-40.0, min(40.0, z))
    return 1.0 / (1.0 + math.exp(-z))


class LogisticRegression:
    def __init__(self, lr=0.05, epochs=900, l2=0.002):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w = [0.0] * len(FEATURES)
        self.b = 0.0
        self.means = [0.0] * len(FEATURES)
        self.scales = [1.0] * len(FEATURES)

    def fit(self, X, y):
        n, m = len(X), len(FEATURES)
        for j in range(m):
            vals = [row[j] for row in X]
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            self.means[j] = mean
            self.scales[j] = math.sqrt(var) or 1.0
        Z = [[(row[j] - self.means[j]) / self.scales[j] for j in range(m)] for row in X]
        for _ in range(self.epochs):
            gw = [0.0] * m
            gb = 0.0
            for row, target in zip(Z, y):
                p = sigmoid(self.b + sum(w * x for w, x in zip(self.w, row)))
                err = p - target
                gb += err
                for j in range(m):
                    gw[j] += err * row[j]
            for j in range(m):
                gw[j] = gw[j] / n + self.l2 * self.w[j]
                self.w[j] -= self.lr * gw[j]
            self.b -= self.lr * gb / n
        return self

    def predict_proba(self, row):
        z = self.b
        for j, x in enumerate(row):
            z += self.w[j] * ((x - self.means[j]) / self.scales[j])
        return sigmoid(z)


def train_demo_model(seed=42):
    """Train on synthetic labelled examples with physically plausible ranges."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(2600):
        rain24 = rng.uniform(5, 260)
        rain72 = min(420, rain24 + rng.uniform(10, 220))
        intensity = rng.uniform(0.5, 55)
        moisture = rng.uniform(15, 98)
        slope = rng.uniform(8, 58)
        elevation = rng.uniform(100, 2200)
        drainage = rng.uniform(0, 1)       # 1 = good drainage
        history = rng.uniform(0, 1)
        tilt = rng.uniform(0, 6)           # mm/day equivalent movement index
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
    return LogisticRegression().fit(X, y)


MODEL = train_demo_model()


def predict(features):
    row = [float(features.get(name, 0.0)) for name in FEATURES]
    probability = MODEL.predict_proba(row)
    return round(probability, 4)


def feature_contributions(features):
    """Return readable contribution weights for the UI, based on model terms."""
    row = [float(features.get(name, 0.0)) for name in FEATURES]
    vals = []
    for name, x, w, mean, scale in zip(FEATURES, row, MODEL.w, MODEL.means, MODEL.scales):
        vals.append((name, w * ((x - mean) / scale)))
    vals.sort(key=lambda x: abs(x[1]), reverse=True)
    labels = {
        "rain24": "24h rainfall",
        "rain72": "72h rainfall",
        "intensity": "rainfall intensity",
        "moisture": "soil moisture",
        "slope_angle": "slope angle",
        "elevation": "elevation",
        "drainage": "drainage condition",
        "history": "past landslide history",
        "tilt_rate": "slope movement",
        "forecast6h": "next 6h rainfall forecast",
    }
    return [{"factor": labels[n], "impact": round(v, 2)} for n, v in vals[:5]]
