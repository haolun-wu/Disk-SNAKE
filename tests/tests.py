"""[WIP]"""
from kbgen.data.datasets import GSM
from kbgen.config import defaults_customLM as config
from unittest import TestCase
from kbgen.model import KBFormer


class TestGSM(TestCase):
    def test_from_config(self):
        dataset = GSM.from_config_(config)


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




def test_padding():
    import os
    from kbgen.utils.log import RunTracker
    from kbgen.config import rootdir
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run = RunTracker.from_logdir(os.path.join(rootdir, "models/09-06-13-05-11longrundecoder-only_l4_d256"))
    run.load_latest_model()
    run.model.eval()
    run.model.to(device)

    out_with_pad = run.model(torch.tensor([[1, 0]]), attention_mask=torch.tensor([[0., float("-inf")]]))
    out_wo_pad = run.model(torch.tensor([[1]]), attention_mask=torch.tensor([[0.]]))
    assert torch.allclose(out_with_pad[0, 0], out_wo_pad[0, 0])

test_padding()
