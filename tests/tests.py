"""[WIP]"""
from kbgen.data.datasets import GSM
from kbgen.config import defaults_customLM as config
from unittest import TestCase
from kbgen.model import KBFormer


class TestGSM(TestCase):
    def test_from_config(self):
        dataset = GSM.from_config(config)


class testKBFormer(TestCase):
    model = KBFormer(config)

    def generate_samples(self):
        num_samples = 10
        input_dict = {}
        # for field in model.config["fields"]:
        #     if field in model.dataset.fields["categorical"]:
        #         input_dict[field] = None
        #     elif field in model.dataset.fields["text"]:
        #         input_dict[field] = None
        #     else:
        #         input_dict[field] = None
        return input_dict

    def test_get_property_mask(self):
        self.model._sample_property_mask(self.generate_samples(), mask_rate=0.5)

    def test_get_predictions(self):
        input_token_dict = self.generate_samples()
        self.model.get_predictions(input_token_dict)
