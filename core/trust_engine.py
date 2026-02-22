class TrustEngine:
    def __init__(self, model, feature_stats):
        self.wrapper = ModelWrapper(model)
        self.feature_stats = feature_stats

    def evaluate(self, X):
        pred, probs = self.wrapper.predict(X)
        p = probs[0][1]

        confidence = probability_confidence(p)
        uncertainty = entropy_uncertainty(p)
        ood_risk = compute_ood(X, self.feature_stats)

        trust = compute_trust(confidence, ood_risk, uncertainty)

        return {
            "prediction": int(pred[0]),
            "confidence": confidence,
            "uncertainty": uncertainty,
            "ood_risk": ood_risk,
            "trust_score": trust
        }
