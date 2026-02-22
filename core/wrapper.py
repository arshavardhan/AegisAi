class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        prediction = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        return prediction, probabilities
