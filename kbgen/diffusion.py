import torch
import torch.nn as nn
from typing import Tuple
from kbgen.utils import TensorDict, like

class HybridDiffusion:
    def __init__(self, model: nn.Module, text_gen_max_len=30) -> None:
        self.model = model
        self.config = model.config
        self.input_shapes = {}
        for field in self.config["fields"]["numerical"]:
            self.input_shapes[field] = [1]
        for field in self.config["fields"]["categorical"]:
            self.input_shapes[field] = [1]
        for field in self.config["fields"]["text"]:
            self.input_shapes[field] = [text_gen_max_len]
        self.numel = len(self.config["fields"].all_fields)

    def _random_mask(self, batch_dim, input_dim, rate=-1, device=None):
        """Generate a random mask for a batch of inputs with probability
        of masking selected uniformly at random from [0, 1].
        """
        if rate == -1:
            mask_rate = torch.rand(batch_dim, 1).repeat(1, input_dim)
        else:
            if not 0 <= rate <= 1:
                raise ValueError("rate must be in [0, 1]")
            mask_rate = torch.ones(batch_dim, input_dim) * rate
        mask = torch.bernoulli(mask_rate).bool()
        return mask.to(device)

    def _flip_n_mask(
        self, mask: torch.Tensor, backward: bool = False, flips: int = 1
    ) -> torch.Tensor:
        """Inplace flip n elements from each row of a batch of masks. When going backward,
        elements are revealed by flipping False to True, and the reverse when going
        forward.

        Args:
            mask (torch.Tensor): (batch, input_dim) mask tensor with True for
                visible elements and False for masked elements.
            backward (bool, optional): Whether diffusion is going backward or forward.
                Defaults to False.
            flips (int, optional): Number of elements to flip. Defaults to 1.

        Returns:
            torch.Tensor: The updated mask tensor.
        """
        mask = mask.clone().view(-1, self.numel)
        for row in mask:
            # during the forward we flip True elements to False
            flippable = ~row if backward else row
            can_flip = min(flippable.sum().item(), flips)
            if can_flip:
                flip_idx = torch.multinomial(flippable.float(), can_flip)
                # set to True if going backward, False if going forward
                row[flip_idx] = backward
        return mask.view(-1, self.numel)

    def _mask_out_batch(
        self, x: torch.Tensor, rate: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask out a batch of inputs for a training step."""
        batch_dim = x.shape[0]
        input_dim = int(torch.prod(torch.tensor(x.shape[1:])).item())
        mask = self._random_mask(batch_dim, input_dim, rate, device=x.device)
        mask = self._flip_n_mask(mask, flips=1)  # flip one additional element
        mask = mask.view(*x.shape)
        return x.masked_fill(mask, 0), mask

    def _bool_mask_to_float(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert a bool mask with True for visible and False for masked
        to a float mask (-inf masked and 0 visible)."""
        mask = mask.clone()
        return mask.logical_not_().float().masked_fill_(mask, -torch.inf)

    def _float_mask_to_bool(self, mask: torch.Tensor) -> torch.Tensor:
        """Convert a float mask with -inf for masked and 0 for visible
        to a bool mask (True for visible and False for masked)."""
        return mask == 0.0

    def _backward_diffusion_step_(
        self,
        x: TensorDict,
        kpm: TensorDict,  # key padding mask needed for text fields
        mask: torch.Tensor,  # property mask
        unmask_idx: torch.Tensor,
        temperature: float = 1,  # temperature for sampling
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # This is where hybrid changes, sample numerical fields with Gaussian
        output = self.model.sample(
            x,
            kpm,
            property_mask=self._bool_mask_to_float(mask),
            temperature=temperature,
            teacher_forcing=False,
        )
        assert len(mask) == len(unmask_idx) # need indices for each sample
        # mask is bool type
        new_mask = mask.scatter(1, unmask_idx, True) # unmask new indices
        diff_mask = new_mask ^ mask # get the difference to avoid overwriting
        # replace the samples in x on diff_mask
        for idx, field in enumerate(self.config["fields"].all_fields):
            to_update = diff_mask[:, idx]
            new_samples = like(output[field], x[field])
            x[field][to_update] = new_samples[to_update]
        return x, new_mask


    def _get_empty_input(self, batch_size: int) -> TensorDict:
        out = {}
        for field in self.config["fields"].all_fields:
            shape = (batch_size, *self.input_shapes[field])
            if field in self.config["fields"]["numerical"]:
                out[field] = torch.zeros(*shape, dtype=torch.float).flatten()
            elif field in self.config["fields"]["categorical"]:
                out[field] = torch.zeros(*shape).long().flatten()
            elif field in self.config["fields"]["text"]:
                out[field] = torch.zeros(*shape).long()
        return TensorDict(out, fields=self.config["fields"])

    def sample(
        self,
        n=1,
        cond=None,
        kpm=None,
        mask=None,
        unmask_order=None,
        leaps: int = 1,
        batch_size : int = 1024,
        temperature: float = 1,
    ):
        device = next(self.model.parameters()).device
        # assumes all parameters are on the same device
        if cond is None:
            assert kpm is None and mask is None and unmask_order is None
            cond = self._get_empty_input(n).to(device)
            mask = torch.zeros(n, self.numel, dtype=torch.bool, device=device)
            kpm = self._get_empty_input(n).float_().to(device)
            unmask_order = torch.rand(n, self.numel).argsort(dim=-1).to(device)
        else:
            assert kpm is not None and mask is not None and unmask_order is not None
        self.model.eval()
        with torch.inference_mode():
            print(f"Sampling {n} samples with batch size {batch_size}, leaps {leaps}, temperature {temperature}")
            for idx in range(0, n, batch_size):
                batch_cond = cond.iloc[idx : idx + batch_size]
                batch_kpm = kpm.iloc[idx : idx + batch_size]
                batch_mask = mask[idx : idx + batch_size]
                batch_unmask_order = unmask_order[idx : idx + batch_size]
                for i in range(0, batch_mask.shape[1], leaps):
                    batch_cond, batch_mask = self._backward_diffusion_step_(
                        batch_cond,
                        batch_kpm,
                        batch_mask,
                        batch_unmask_order[:, i : i + leaps],
                        temperature=temperature,
                    )
                cond.iloc[idx : idx + batch_size] = batch_cond
                mask[idx : idx + batch_size] = batch_mask
        return cond

__all__ = ["HybridDiffusion"]
