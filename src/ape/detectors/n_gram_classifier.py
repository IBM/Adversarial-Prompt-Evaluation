import pickle

from ape.detectors.detector import Detector


class N_gram_classifier(Detector):
    """
    Tabular classifier with n-gram feature extraction.
    """

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.path = path
        self.model = pickle.load(open(self.path, "rb"))        

    def predict(self, prompts):
        prompts_pred_probs = self.predict_proba(prompts)

        prompts_preds = [
            (
                {"label": "unsafe", "confidence": p[1]}
                if p[1] > p[0]
                else {"label": "safe", "confidence": p[0]}
            )
            for p in prompts_pred_probs
        ]

        return prompts_preds

    def predict_proba(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        return self.model.predict_proba(prompts)
