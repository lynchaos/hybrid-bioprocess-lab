"""PyTorch implementation of the CorrectionModel protocol.

The point of this file is to prove that the CorrectionModel abstraction is
real, not decorative. Swapping the sklearn MLP for a Torch module should be a
new file here and nothing else in the hybrid or mechanistic code.

Because this correction is called inside an ODE right-hand side, smoothness
still matters. We use a small MLP with smooth activations (SiLU) and keep the
forward pass pure: no batch norm, no dropout, no stochasticity.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.optim import Adam

Array = NDArray[np.float64]


class _CorrectionNet(nn.Module):
    """Small, smooth MLP that predicts a bounded growth-rate multiplier."""

    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...],
        lower: float,
        upper: float,
    ) -> None:
        super().__init__()
        if lower <= 0.0:
            raise ValueError("lower bound must be > 0")
        if lower >= upper:
            raise ValueError(f"invalid bounds: ({lower}, {upper})")
        self._lower = float(lower)
        self._upper = float(upper)

        dims = [input_dim, *hidden, 1]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=True):
            layers.extend([nn.Linear(in_dim, out_dim), nn.SiLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).squeeze(-1)
        # Squash to (lower, upper) with a sigmoid. Differentiable and bounded.
        return self._lower + (self._upper - self._lower) * torch.sigmoid(raw)


class TorchCorrection:
    """A CorrectionModel backed by a tiny Torch MLP.

    This is deliberately *not* a general-purpose neural-network trainer. It is
    a production-grade replacement for the sklearn correction, with the same
    contract: fit on (X, y), predict bounded multipliers, refuse before fit.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (16,),
        bounds: tuple[float, float] = (0.5, 1.8),
        epochs: int = 500,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.bounds = bounds
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self._net: _CorrectionNet | None = None
        self._scaler_mean: Array | None = None
        self._scaler_std: Array | None = None

    def fit(self, X: Array, y: Array) -> TorchCorrection:
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        if len(X) == 0:
            raise ValueError("refusing to fit on zero rows")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._scaler_mean = np.mean(X, axis=0)
        self._scaler_std = np.std(X, axis=0)
        self._scaler_std = np.where(self._scaler_std < 1e-8, 1.0, self._scaler_std)
        Xs = (X - self._scaler_mean) / self._scaler_std

        self._net = _CorrectionNet(
            input_dim=X.shape[1],
            hidden=self.hidden,
            lower=self.bounds[0],
            upper=self.bounds[1],
        ).double()
        self._net.train()

        Xt = torch.tensor(Xs, dtype=torch.float64)
        yt = torch.tensor(y, dtype=torch.float64)
        optimizer = Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self._net(Xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            optimizer.step()

        self._net.eval()
        return self

    def predict(self, X: Array) -> Array:
        if self._net is None or self._scaler_mean is None or self._scaler_std is None:
            raise RuntimeError(
                "TorchCorrection.predict called before fit. Refusing to return "
                "an identity correction, which would silently degrade this to a "
                "mechanistic model while still reporting as hybrid."
            )
        self._net.eval()
        with torch.no_grad():
            Xs = (X - self._scaler_mean) / self._scaler_std
            Xt = torch.tensor(Xs, dtype=torch.float64)
            pred = self._net(Xt).numpy()
        return np.asarray(pred, dtype=np.float64)

    @property
    def is_fitted(self) -> bool:
        return self._net is not None

    def save(self, path: Path) -> Path:
        """Serialize the fitted network and its scaler."""
        path = Path(path)
        if not self.is_fitted:
            raise RuntimeError("cannot save an unfitted TorchCorrection")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "state_dict": self._net.state_dict(),  # type: ignore[union-attr]
                "scaler_mean": self._scaler_mean,
                "scaler_std": self._scaler_std,
                "hidden": self.hidden,
                "bounds": self.bounds,
                "epochs": self.epochs,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "seed": self.seed,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> TorchCorrection:
        """Load a previously saved TorchCorrection."""
        payload = joblib.load(path)
        obj = cls(
            hidden=tuple(payload["hidden"]),
            bounds=tuple(payload["bounds"]),
            epochs=int(payload["epochs"]),
            lr=float(payload["lr"]),
            weight_decay=float(payload["weight_decay"]),
            seed=int(payload["seed"]),
        )
        obj._scaler_mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
        obj._scaler_std = np.asarray(payload["scaler_std"], dtype=np.float64)
        obj._net = _CorrectionNet(
            input_dim=obj._scaler_mean.shape[0],
            hidden=obj.hidden,
            lower=obj.bounds[0],
            upper=obj.bounds[1],
        ).double()
        obj._net.load_state_dict(payload["state_dict"])
        obj._net.eval()
        return obj


def torch_estimator(
    hidden: tuple[int, ...] = (16,),
    bounds: tuple[float, float] = (0.5, 1.8),
    epochs: int = 500,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    seed: int = 0,
) -> TorchCorrection:
    """Factory for the default Torch-backed correction."""
    return TorchCorrection(
        hidden=hidden,
        bounds=bounds,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
    )
