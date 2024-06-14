import requests
from typing import List, Dict
from ape.detectors.detector import Detector


class AzureAPI(Detector):
    """
    Azure API for jailbreak detection
    """
    def __init__(self, endpoint: str = "https://runazure.cognitiveservices.azure.com", subscription_key: str = None):
        """
        Parameters
        ----------
        endpoint : str
        subscription_key : str
        """
        super().__init__()
        self.endpoint = f"{endpoint}/contentsafety/text:detectJailbreak?api-version=2023-10-15-preview"
        self.__subscription_key = subscription_key
        self.__headers = {"Ocp-Apim-Subscription-Key": self.__subscription_key, "Content-Type": "application/json"}
        self.data_function = lambda prompt: {"text": prompt}

    def predict(self, prompts: List[str], **kwarg) -> List[Dict]:
        """
        Parameters
        ----------
        prompts : List[str]

        Returns
        -------
        results: List[dict]
        """
        results = []
        for prompt in prompts:
            prompt_format = self.data_function(prompt)
            response = requests.post(self.endpoint, headers=self.__headers, json=prompt_format).json()
            if response["jailbreakAnalysis"]["detected"]:
                results.append({"label": "unsafe", "confidence": None, "api_response": response})
            else:
                results.append({"label": "safe", "confidence": None, "api_response": response})
        return results

    def predict_proba(self, sample):
        raise NotImplementedError
