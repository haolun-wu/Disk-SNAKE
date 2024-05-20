import torch
from kbgen.utils import AggregatedMetrics, ModelOutputs, Fields
from collections import defaultdict


def test_aggregated_metrics():
    # Define some example configuration
    config = {
        "fields": Fields(numerical=["num1", "num2"], categorical=["cat1", "cat2"]),
        "numerical_pad_token_id": 0,
        "categorical_pad_token_id": 0,
    }

    # Create an instance of the AggregatedMetrics class
    agg_metrics = AggregatedMetrics(config)

    # Define some example model outputs
    num_samples = 4
    # 0 is the pad token ID so these do not contribute to the loss
    targets = {
        "num1": torch.tensor([0, 2, 3, 4], dtype=torch.float32),
        "num2": torch.tensor([5, 6, 7, 8], dtype=torch.float32),
        "cat1": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "cat2": torch.tensor([5, 6, 7, 8], dtype=torch.long),
    }
    property_mask = torch.tensor(
        [
            [True, False, False, False],
            [True, False, False, True],
            [True, False, False, True],
            [True, False, True, True],
        ],
        dtype=torch.float32,
    )
    property_mask.masked_fill_(property_mask.bool(), -torch.inf)

    loss_dict = {
        "num1": torch.tensor(1.0),
        "num2": torch.tensor(2.0),
        "cat1": torch.tensor(3.0),
        "cat2": torch.tensor(4.0),
    }
    masked_loss_dict = {
        "num1": torch.tensor(1.5),
        "num2": torch.tensor(2.5),
        "cat1": torch.tensor(3.5),
        "cat2": torch.tensor(4.5),
    }
    unmasked_loss_dict = {
        "num1": torch.tensor(2.0),
        "num2": torch.tensor(3.0),
        "cat1": torch.tensor(4.0),
        "cat2": torch.tensor(5.0),
    }
    masked_error_dict = {
        "num1": torch.tensor(0.1),
        "num2": torch.tensor(0.2),
        "cat1": torch.tensor(0.3),
        "cat2": torch.tensor(0.4),
    }
    unmasked_error_dict = {
        "num1": torch.tensor(0.2),
        "num2": torch.tensor(0.3),
        "cat1": torch.tensor(0.4),
        "cat2": torch.tensor(0.5),
    }
    model_outputs = ModelOutputs(
        preds=targets,
        targets=targets,
        property_mask=property_mask,
        loss=sum(loss_dict.values()),
        loss_dict=loss_dict,
        masked_loss_dict=masked_loss_dict,
        unmasked_loss_dict=unmasked_loss_dict,
        masked_error_dict=masked_error_dict,
        unmasked_error_dict=unmasked_error_dict,
    )

    # Add the model outputs to the aggregated metrics
    agg_metrics.add_contribution(model_outputs)

    # Check that the aggregated metrics have been updated correctly
    assert agg_metrics.num_field_samples == {"num1": 3, "num2": 4, "cat1": 4, "cat2": 4}
    assert agg_metrics.num_masked_field_samples == {
        "num1": 3,
        "num2": 0,
        "cat1": 1,
        "cat2": 3,
    }
    assert agg_metrics.num_unmasked_field_samples == {
        "num1": 0,
        "num2": 4,
        "cat1": 3,
        "cat2": 1,
    }
    assert agg_metrics.loss_dict == {"num1": 1.0, "num2": 2.0, "cat1": 3.0, "cat2": 4.0}
    assert agg_metrics.masked_loss_dict == {
        "num1": 1.5,
        "num2": 0.0,
        "cat1": 3.5,
        "cat2": 4.5,
    }
    assert agg_metrics.unmasked_loss_dict == {
        "num1": 0.0,
        "num2": 3.0,
        "cat1": 4.0,
        "cat2": 5.0,
    }
    assert agg_metrics.masked_error_dict == {
        "num1": 0.1,
        "num2": 0.0,
        "cat1": 0.3,
        "cat2": 0.4,
    }
    assert agg_metrics.unmasked_error_dict == {
        "num1": 0.0,
        "num2": 0.3,
        "cat1": 0.4,
        "cat2": 0.5,
    }

    # make more model outputs

    property_mask_2 = torch.tensor(
        [
            [False, True, False, True],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ],
        dtype=torch.float32,
    )
    property_mask_2.masked_fill_(property_mask_2.bool(), -torch.inf)

    model_outputs_2 = ModelOutputs(
        preds=targets,
        targets=targets,
        property_mask=property_mask_2,
        loss=sum(loss_dict.values()),
        loss_dict={k: v + 1 for k, v in loss_dict.items()},
        masked_loss_dict={k: v + 1 for k, v in masked_loss_dict.items()},
        unmasked_loss_dict={k: v + 1 for k, v in unmasked_loss_dict.items()},
        masked_error_dict={k: v + 1 for k, v in masked_error_dict.items()},
        unmasked_error_dict={k: v + 1 for k, v in unmasked_error_dict.items()},
    )

    # Add the model outputs to the aggregated metrics
    agg_metrics.add_contribution(model_outputs_2)

    # Check that the aggregated metrics have been updated correctly
    assert agg_metrics.num_field_samples == {
        "num1": 3 + 3,
        "num2": 4 + 4,
        "cat1": 4 + 4,
        "cat2": 4 + 4,
    }
    assert agg_metrics.num_masked_field_samples == {
        "num1": 0 + 3,
        "num2": 1 + 2,
        "cat1": 1 + 2,
        "cat2": 3 + 1,
    }
    assert agg_metrics.num_unmasked_field_samples == {
        "num1": 0 + 3,
        "num2": 4 + 1,
        "cat1": 3 + 2,
        "cat2": 1 + 3,
    }
    assert agg_metrics.loss_dict == {
        "num1": ((1 * 3) + (2 * 3)) / 6,
        "num2": 2.5,
        "cat1": 3.5,
        "cat2": 4.5,
    }
    assert agg_metrics.masked_loss_dict == {
        "num1": 1.5,
        "num2": 0.0 + (2.5 + 1),
        "cat1": (3.5 * 1 + (3.5 + 1) * 2) / 3,
        "cat2": (4.5 * 3 + (5.5) * 1) / 4,
    }
    assert agg_metrics.unmasked_loss_dict == {
        "num1": 0.0 + 3,
        "num2": (3.0 * 4 + 4.0 * 1) / 5,
        "cat1": (4.0 * 3 + 5 * 2) / 5,
        "cat2": (5.0 * 1 + 6.0 * 3) / 4,
    }
    assert agg_metrics.masked_error_dict == {
        "num1": 0.1,
        "num2": 1.2,
        "cat1": (0.3 * 1 + 1.3 * 2) / 3,
        "cat2": (0.4 * 3 + 1.4 * 1) / 4,
    }
    assert agg_metrics.unmasked_error_dict == {
        "num1": 0.0 + 1.2,
        "num2": (0.3 * 4 + 1.3) / 5,
        "cat1": (0.4 * 3 + 1.4 * 2) / 5,
        "cat2": (0.5 * 1 + 1.5 * 3) / 4,
    }
    print("Finished successfully")

test_aggregated_metrics()
