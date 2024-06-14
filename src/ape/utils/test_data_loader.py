"""
Helper functions to aid in asserting for file loading and consistency.
"""

import hashlib
import json
from typing import Dict


class TestLoader:
    """
    Class to streamline the loading and checking of the test set for benchmarking
    """

    expected_hashes = {
        "expected_hash_all": "a5f4372c526e348a6bc37b0f13d5b6112deacc264208943d91094d13749b3344",
        "expected_hash_test": "7cbce2c82e52d2c4229d9b48a7f19a3f8390e06ea6722fe4b19ea98aa9ee3c23",
        "expected_hash_sub_sample_filtered": "796be1e2d5c4c2a8ef1758531674bc3864b001eaf27cd64a9eee5a6b932152b3",
        "expected_hash_sub_sample": "bd96669bdefa0a0223940182c44aae298fc43b0bff102fa0803fa794bb09d423",
        "expected_hash_ood": "400a4a3fe8637be4f124324ffa3f2595a6f792ccdd99cda5889915ae0a79f7ca",
        "expected_hash_ood_filtered": "09fcabf22ee6a5b8a4c3694f0cf1a71a5269b14b69f24ffffbace6c57bbdad85",
        "expected_hash_risk_eval": "d455dcf6b8d25ce2dc7d9cf99f35452b111cace0157c1220de800ab2fe2845f4",
    }

    @staticmethod
    def hash_file(filename: str) -> str:
        """
        Computes the sha256 hash of the saved file to check that the dataset is consistent between runs.

        :param filename: file to compute the sha256 hash of

        :return: sha256 hash of the indicated file
        """

        h = hashlib.sha256()

        with open(filename, "rb") as file:
            chunk = 0
            while chunk != b"":
                chunk = file.read(1024)
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def load_test_set(cls, filename: str, data_type: str) -> Dict:
        """
        Handles loading the data, checking it, and returns it in a consumable format

        :param filename: json file to load
        :param data_type: the type of data to load (one of "all", "sub_sample", "ood", "risk_eval").
        :return: dictionary with the test set data
        """
        if data_type not in ["all", "sub_sample", "sub_sample_filtered", "ood", "ood_filtered", "risk_eval"]:
            raise ValueError(f"{data_type} data_type invalid for hash selection")

        computed_hash = cls.hash_file(filename)

        print(f"Computed hash {computed_hash}")

        expected_hash = cls.expected_hashes["expected_hash_" + data_type]

        if computed_hash != expected_hash:
            print(
                f"\033[1;31mHash for {data_type} json does not match! If this is not expected, "
                "check the datasets being loaded and samples fetched\033[0;0m"
            )

        with open(filename, encoding="utf-8") as f:
            data = json.load(f)

        output = {"prompt": [], "source": [], "label": []}
        for sample in data:
            for tag in ["prompt", "source", "label"]:
                output[tag].append(sample[tag])

        return output
