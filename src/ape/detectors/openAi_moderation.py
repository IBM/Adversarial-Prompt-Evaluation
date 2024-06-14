import os

from typing import Union, List

from openai import OpenAI


class OpenAIModeration:
    def __init__(self, token):
        os.environ["OPENAI_API_KEY"] = token
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @staticmethod
    def filter_result(result):
        prediction = int(result.results[0].flagged)
        probabilities = list(result.results[0].category_scores)
        proba = max([i[1] for i in probabilities])
        return prediction, proba, probabilities

    def predict(self, prompt: Union[str, List[str]]) -> List[dict]:
        if isinstance(prompt, str):
            prompt = [prompt]
        else:
            prompts_pred_probs = []
            for p in prompt:
                results = self.predict_proba(p)
                pred, proba, probabilities = self.filter_result(results)
                prompts_pred_probs.append([1-proba, proba])
            prompts_preds = [
                (
                    {"label": "unsafe", "confidence": p[1]}
                    if p[1] > p[0]
                    else {"label": "safe", "confidence": p[0]}
                )
                for p in prompts_pred_probs
            ]
            return prompts_preds

    def predict_proba(self, prompt: str):
        return self.client.moderations.create(input=prompt)
