"""
This file is used to load and run a trained model. for evaluation purposes.
"""
import torch
from kbgen.config import rootdir
from kbgen.data.datasets import GSM
from kbgen.utils.log import RunTracker
from kbgen.utils import TensorDict
import os


class Demo:
    def load_run(
        self,
        model_name=None,
        model_path=None,
        force_device=None,
    ):
        assert (
            model_name is not None or model_path is not None
        ), "You need to specify either a model name or a model path."
        
        # DATA -----------------------------------------------------
        device = force_device if force_device is not None else "cuda:0"
        torch.set_default_device(device)
        # Load Wandb Run -----------------------------------------------------
        if model_path is not None:
            logdir = model_path
        else:
            logdir = os.path.join(rootdir, "models", model_name)
        run = RunTracker.from_logdir(logdir, force_device=device)
        dataset = GSM(
            os.path.join(rootdir, "data/gsm"),
            tokenizer=run.config.tokenizer,
        )
        assert (
            run.config.fields == dataset.fields
        ), "Fields do not match. Check that the dataset is the same as the one used for training."
        model = run.load_latest_model().to(device)
        model.eval()
        print(run.config)
        self.model = model
        self.dataset = dataset
        self.run = run
        return model, dataset, run

    def sample(self, num, input_dict, mask_none=False, temp=0.0, resample_given=False):
        """
        This method is used to sample data from the dataset.

        Args:
            num (int): The number of samples to be drawn from the dataset.
            input_dict (dict): A dictionary containing the data we want to give as input to the model.
                The keys are the field names and the values are the corresponding data.
            mask_none (bool, optional): If True, the fields with None values in the input_dict will be masked.
                Otherwise, they will be resampled. Defaults to False.
            temp (int, optional): The temperature parameter for the sampling process.
                Higher values make the sampling more random, lower values make it more deterministic.
                Defaults to 0.
            resample_given (bool, optional): If True, the fields with given values in the input_dict will
                be resampled. Defaults to False.

        Returns:
            dict: A dictionary containing the sampled data. The keys are the field names and the values are the corresponding sampled data.
        """
        input_dict = input_dict.copy()
        if not isinstance(input_dict, TensorDict):
            input_dict = TensorDict(input_dict, fields=self.dataset.fields)
        indices = torch.randint(0, len(self.dataset), (num,))
        tensor_dict = self.dataset[indices]
        tensor_dict = {field: tensor.clone() for field, tensor in tensor_dict.items()}
        tensor_dict = TensorDict(tensor_dict, fields=self.dataset.fields)
        for field, value in input_dict.items():
            if value is not None:
                if (
                    field in self.dataset.fields["categorical"]
                    or field in self.dataset.fields["text"]
                ):
                    value = self.dataset.tokenizer.encode(value)
                    value = torch.tensor(value).repeat(num, 1)
                else:
                    value = torch.tensor(value).repeat(num)
                tensor_dict[field] = value
        if mask_none:
            # p1 is not none, p2 is none
            # {p1: [0, 0, 0], p2: [-inf, -inf, -inf]}
            property_mask = {}
            for k in input_dict:
                mask_value = -torch.inf if input_dict[k] is None else 0.0
                property_mask[k] = torch.full((num,), mask_value)
        else:
            property_mask = None
        with torch.no_grad():
            preds = self.model.get_predictions(tensor_dict, property_mask=property_mask)
        temps = {
            field: temp if (input_dict[field] is None or resample_given) else 0
            for field in preds
        }
        result = {}
        for field in preds:
            if field in self.dataset.fields["categorical"]:
                sample_classes = (
                    sample_with_temp(preds[field], temps[field]).view(-1).tolist()
                )
                result[field] = self.dataset.label_to_string(sample_classes, field)
            elif field in self.dataset.fields["text"]:
                sample_tokens = sample_with_temp(preds[field], temps[field])
                result[field] = self.dataset.tokenizer.batch_decode(
                    sample_tokens, skip_special_tokens=True
                )

            elif field in self.dataset.fields["numerical"]:
                sample_num = sample_gaussian(preds[field].flatten(), temps[field])
                result[field] = self.dataset.reverse_transform({field: sample_num})[
                    field
                ].tolist()

        return Result(result)

    def print_fields(self):
        for field_type in self.dataset.fields:
            print("*", field_type)
            if field_type == "numerical":
                print(
                    *[
                        "\t- "
                        + f"{f:<20}"
                        + f"[{self.dataset._numerical_min[f]}, {self.dataset._numerical_max[f]}]"
                        for f in self.dataset.fields[field_type]
                    ],
                    sep="\n",
                )
            else:
                print(
                    *["\t- " + f for f in self.dataset.fields[field_type]],
                    sep="\n",
                )


def sample_with_temp(logits, temp):
    if temp == 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temp, dim=-1)
    shape = probs.shape
    probs = probs.view(-1, probs.shape[-1])
    out = torch.multinomial(probs, 1)
    return out.view(shape[:-1])


def sample_gaussian(mu, sigma):
    sigma = (torch.full_like(mu, sigma).exp() - 1) / 100
    return torch.normal(
        mu,
        sigma,
    )


class Result:
    def __init__(self, result_dict):
        self.result_dict = result_dict

    def __repr__(self):
        join_value_lists = (
            lambda x: ", ".join([str(v) for v in x]) if isinstance(x, list) else x
        )
        return "\n".join(
            [f"{k:<20} {join_value_lists(v)}" for k, v in self.result_dict.items()]
        )

    def __getitem__(self, key):
        return self.result_dict[key]

    def keys(self):
        return self.result_dict.keys()

    def values(self):
        return self.result_dict.values()


__all__ = ["Demo"]
