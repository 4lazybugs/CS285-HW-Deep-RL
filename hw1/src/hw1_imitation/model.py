"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim * chunk_size))
        self.net = nn.Sequential(*layers)


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [B, state_dim]
        returns predicted action chunk: [B, chunk_size, action_dim]
        """
        out = self.net(state)
        return out.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        pred = self.forward(state)  # [B, chunk_size, action_dim]
        loss = nn.functional.mse_loss(pred, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # MSE policy is deterministic — num_steps is unused, just returns forward().
        return self.forward(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        in_dim = state_dim + action_dim * chunk_size + 1
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim * chunk_size))
        self.net = nn.Sequential(*layers)

    def forward(
            self, 
            state: torch.Tensor, 
            action_chunk: torch.Tensor,
            tau: torch.Tensor
            ) -> torch.Tensor:
        """
        state: [B, state_dim]
        action_chunk: [B, chunk_size, action_dim]  (this is A_tau, the noisy/interpolated chunk)
        tau: [B, 1]  (flow matching timestep, one scalar per batch element)
        returns predicted velocity: [B, chunk_size, action_dim]
        """
        B = state.shape[0]
        flat_action = action_chunk.reshape(B, -1)  # [B, chunk_size * action_dim]
        input_tensor = torch.cat([state, flat_action, tau], dim=1)
        out = self.net(input_tensor)
        return out.view(B, self.chunk_size, self.action_dim)
    
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        device = state.device

        # A_0 ~ N(0, I), same shape as action_chunk
        A_0 = torch.randn_like(action_chunk)
        A = action_chunk

        # tau ~ U(0, 1), one scalar per sample in the batch
        tau = torch.rand(B, 1, device=device)  # [B, 1]
        # reshape for broadcasting against [B, chunk_size, action_dim]
        tau_broadcast = tau.view(B, 1, 1)

        # interpolate: A_tau = tau * A + (1 - tau) * A_0
        A_tau = tau_broadcast * A + (1 - tau_broadcast) * A_0

        # predict velocity, conditioned on state, A_tau, and tau
        vel_pred = self.forward(state, A_tau, tau)  # [B, chunk_size, action_dim]

        # target velocity: A - A_0
        vel_target = A - A_0

        loss = nn.functional.mse_loss(vel_pred, vel_target)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        device = state.device

        # start from pure noise: A_{t,0} ~ N(0, I)
        A_tau = torch.randn(B, self.chunk_size, self.action_dim, device=device)

        dt = 1.0 / num_steps
        for n in range(num_steps):
            tau_val = n * dt
            tau = torch.full((B, 1), tau_val, device=device)
            vel_pred = self.forward(state, A_tau, tau)
            A_tau = A_tau + dt * vel_pred

        return A_tau


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
