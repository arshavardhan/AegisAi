def compute_trust(confidence, ood_risk, uncertainty):
    return (
        0.5 * confidence +
        0.3 * (1 - ood_risk) +
        0.2 * (1 - uncertainty)
    )
