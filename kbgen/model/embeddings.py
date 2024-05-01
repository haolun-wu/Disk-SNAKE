import torch
import torch.nn as nn


class DICE(torch.nn.Module):
    """
    DICE class turns numbers into their respective DICE embeddings
    Since the cosine function decreases monotonically between 0 and pi,
    simply employ a linear mapping
    to map distances s_n in [0, |a-b|] to angles in [0, pi]
    """

    def __init__(self, d=2, min_bound=0, max_bound=100, r=1):
        super().__init__()
        if d < 2:
            raise ValueError(
                "Wrong value for `d`. `d` should be greater than or equal to 2."
            )
        self.d = d  # By default, we build DICE-2
        self.r = r
        self.min_bound = min_bound
        self.max_bound = max_bound
        rng = torch.Generator().manual_seed(42)
        M = torch.randn((self.d, self.d), dtype=torch.float64, generator=rng)
        # QR decomposition for orthonormal basis, Q
        Q, _ = torch.linalg.qr(M, mode="complete")
        self.register_buffer("Q", Q)
        sin = torch.arange(0, self.d)
        self.register_buffer("sin_powers", sin, persistent=False)
        self.sin_powers[-2] = d - 1
        self.sin_powers[-1] = d - 2
        cos = torch.ones_like(sin)
        self.register_buffer("cos_powers", cos, persistent=False)
        self.cos_powers = torch.ones_like(self.sin_powers)
        self.cos_powers[-2] = 0
        self.Q: torch.Tensor

    def _linear_mapping(self, nums):
        """Eq. (4) from DICE"""
        nums = (nums - self.min_bound) / (self.max_bound - self.min_bound)
        nums = nums.clamp(0, 1)
        theta = nums * torch.pi
        return theta.double()

    def forward(self, nums):
        if not isinstance(nums, torch.Tensor):
            nums = torch.tensor(nums)
        theta = self._linear_mapping(nums)
        sin = torch.sin(theta).unsqueeze(-1)
        cos = torch.cos(theta).unsqueeze(-1)
        polar_coord = (sin**self.sin_powers) * (cos**self.cos_powers)
        polar_coord *= self.r
        # DICE-D embedding for `num`
        dice = polar_coord @ self.Q
        return dice


class Embedding(nn.Embedding):
    def unembed(self, x):
        # x: (batch_size, d_model)
        x = x @ self.weight.T
        return x


class DiceEmbedding(nn.Module):
    def __init__(
        self, d_model, min_bound=-1000, max_bound=2100, r=1, affine=True
    ) -> None:
        super().__init__()
        self._dice = DICE(d_model, min_bound=min_bound, max_bound=max_bound, r=r)
        self.weight = nn.Parameter(torch.ones(1), requires_grad=affine)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=affine)

    def forward(self, x):
        x = (x - self.bias) * self.weight
        embeddings = self._dice(x)
        return embeddings.to(torch.get_default_dtype())


class PeriodicEmbedding(nn.Embedding):
    def __init__(self, d_model, n_freq=4):
        if n_freq is None:
            n_freq = d_model
        super().__init__(1, n_freq)
        self.projection = nn.Linear(n_freq, d_model)

    def forward(self, x):
        freq = self.weight.sigmoid()
        sin = torch.sin(x.unsqueeze(-1) * freq[:, ::2])
        cos = torch.cos(x.unsqueeze(-1) * freq[:, 1::2])
        return self.projection(torch.cat([sin, cos], dim=-1))


class BinnedEmbedding(nn.Module):
    def __init__(self, d_model, extent=(0, 2100)):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.bias = nn.Parameter(torch.linspace(extent[0], extent[1], d_model))

    def forward(self, x):
        coef = self.weight.exp2()
        return ((x.unsqueeze(-1) - self.bias) * coef).tanh()


class NumericEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config["num_emb"].lower() == "dice":
            embedding = DiceEmbedding(
                config["d_model"],
                min_bound=-1,
                max_bound=2,
                affine=False,
            )
        elif config["num_emb"].lower() == "periodic":
            embedding = PeriodicEmbedding(d_model=config["d_model"])
        elif config["num_emb"].lower() == "binned":
            embedding = BinnedEmbedding(
                d_model=config["d_model"],
            )
        else:
            raise ValueError(f"Unknown numerical embedding type {config['num_emb']}")
        self.embedding = embedding

    def forward(self, x):
        return self.embedding(x)
