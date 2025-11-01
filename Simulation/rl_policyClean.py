import sys
import torch
import torch.nn as nn

# Ensure we can import Sample Factory utilities if needed
sys.path.insert(0, 'sample-factory')


EPS = 1e-5  # Match Sample Factory's _NORM_EPS from running_mean_std.py


class RLPolicyClean(nn.Module):
    """Minimal, inference-only policy that mirrors Sample Factory forward pass.

    Components:
      - Input normalization using checkpoint running mean/var
      - Encoder MLP (Linear/ReLU × 3) for vector obs
      - GRU core
      - Linear distribution head producing [mean, logstd]

    Forward returns (action_logits, new_hxs) where action_logits = [mean, logstd].
    """

    def __init__(self):
        super().__init__()
        self.obs_mean: torch.Tensor | None = None
        self.obs_var: torch.Tensor | None = None
        self.encoder: nn.Module | None = None
        self.core: nn.GRU | None = None
        self.decoder: nn.Module | None = None  # kept for parity; identity by default
        self.dist_linear: nn.Linear | None = None
        # Internal recurrent state (not exposed to user)
        self.hxs: torch.Tensor | None = None  # shape [num_layers, batch, hidden]

    def forward(self, obs: torch.Tensor, *, normalized: bool = False):
        # Normalize exactly like SF when inputs are raw
        obs = torch.tensor(obs, dtype=torch.float32)
        # Ensure observation has batch dimension [batch, features]
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # [features] -> [1, features]
        if not normalized:
            x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + EPS)
            x = torch.clamp(x, -5.0, 5.0)
        else:
            x = obs

        # Encoder MLP expects [batch, features]
        x = self.encoder(x)  # [batch, encoder_out]

        # GRU expects [seq_len, batch, features]; hxs is [num_layers, batch, hidden]
        x = x.unsqueeze(0)  # [batch, features] -> [1, batch, features]
        if self.hxs is None or self.hxs.shape[1] != x.shape[1]:
            self.hxs = torch.zeros(1, x.shape[1], self.core.hidden_size, device=x.device)
        x, h_next = self.core(x, self.hxs)
        x = x.squeeze(0)  # [1, batch, hidden] -> [batch, hidden]
        # Keep internal hidden state for next call
        self.hxs = h_next

        # Decoder (identity by default)
        if self.decoder is not None:
            x = self.decoder(x)

        # Distribution parameters [mean, logstd]
        params = self.dist_linear(x)
        mean, logstd = params.chunk(2, dim=-1)
        action_logits = torch.cat([mean, logstd], dim=-1)
        return action_logits, h_next

    def _init_modules(self, obs_mean: torch.Tensor, obs_var: torch.Tensor,
                       encoder: nn.Module, core: nn.GRU, dist_linear: nn.Linear,
                       decoder: nn.Module | None = None):
        self.obs_mean = nn.Parameter(obs_mean.float(), requires_grad=False)
        self.obs_var = nn.Parameter(obs_var.float(), requires_grad=False)
        self.encoder = encoder
        self.core = core
        self.decoder = decoder
        self.dist_linear = dist_linear
        self.hxs = None

    def reset_hidden_state(self, batch_size: int = 1, device: torch.device | None = None) -> None:
        """Reset internal RNN state to zeros for the given batch size."""
        if device is None:
            device = next(self.parameters()).device
        self.hxs = torch.zeros(1, batch_size, self.core.hidden_size, device=device)

    def set_hidden_state(self, hxs: torch.Tensor) -> None:
        """Set internal RNN state to match external state.
        
        Args:
            hxs: Hidden state tensor of shape [batch, hidden_size] or [1, batch, hidden_size]
        """
        if hxs.dim() == 2:
            # Convert [batch, hidden_size] to [1, batch, hidden_size]
            self.hxs = hxs.unsqueeze(0).clone()
        else:
            # Already [1, batch, hidden_size] or [num_layers, batch, hidden_size]
            self.hxs = hxs.clone()

    @staticmethod
    def _activation(nonlinearity: str) -> nn.Module:
        if nonlinearity == 'relu':
            return nn.ReLU(inplace=False)
        if nonlinearity == 'elu':
            return nn.ELU(inplace=False)
        if nonlinearity == 'tanh':
            return nn.Tanh()
        raise RuntimeError(f"Unsupported nonlinearity: {nonlinearity}")

    @staticmethod
    def _find_mlp_linear_indices(ckpt: dict) -> list[int]:
        """Find ordered Linear layer indices inside encoder.encoders.obs.mlp_head.*
        Sample Factory create_mlp uses indices 0,2,4,... for Linear layers (activations in between).
        """
        prefix = 'encoder.encoders.obs.mlp_head.'
        linear_indices: set[int] = set()
        for k in ckpt.keys():
            if not k.startswith(prefix):
                continue
            if k.endswith('.weight'):
                try:
                    idx = int(k[len(prefix):].split('.')[0])
                except Exception:
                    continue
                linear_indices.add(idx)
        return sorted(linear_indices)

    @classmethod
    def _build_encoder_from_ckpt(cls, ckpt: dict, *, nonlinearity: str, jit: bool) -> nn.Module:
        act = cls._activation(nonlinearity)
        indices = cls._find_mlp_linear_indices(ckpt)
        if len(indices) == 0:
            # No MLP layers configured
            enc = nn.Identity()
            return enc

        # Construct layers in the exact order of indices
        layers: list[nn.Module] = []
        for i, idx in enumerate(indices):
            w_key = f'encoder.encoders.obs.mlp_head.{idx}.weight'
            b_key = f'encoder.encoders.obs.mlp_head.{idx}.bias'
            weight = ckpt[w_key]
            bias = ckpt[b_key]
            in_f, out_f = weight.shape[1], weight.shape[0]
            lin = nn.Linear(in_f, out_f)
            # Load weights immediately to avoid dtype/device surprises later
            lin.weight.data.copy_(weight)
            lin.bias.data.copy_(bias)
            layers.append(lin)
            layers.append(act)

        enc = nn.Sequential(*layers)
        if jit:
            enc = torch.jit.script(enc)
        return enc

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str,
        device: str = 'cpu',
        *,
        nonlinearity: str = 'relu',
        jit_encoder: bool = False,
    ) -> 'RLPolicyClean':
        """Create a policy instance from a Sample Factory checkpoint (.pth).

        Args:
            path: checkpoint path (best_*.pth or checkpoint_*.pth)
            device: 'cpu' or 'cuda'
            nonlinearity: 'relu' | 'elu' | 'tanh' — must match training config
            jit_encoder: if True, torch.jit.script the encoder MLP
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)["model"]

        # Observation normalizer parameters (RunningMeanStd)
        mean_key = 'obs_normalizer.running_mean_std.running_mean_std.obs.running_mean'
        var_key = 'obs_normalizer.running_mean_std.running_mean_std.obs.running_var'
        if mean_key in ckpt and var_key in ckpt:
            obs_mean = ckpt[mean_key].detach()
            obs_var = ckpt[var_key].detach()
        else:
            # Fallback: infer input size from the first linear layer if present, otherwise 1
            indices = [0]
            try:
                indices = RLPolicyClean._find_mlp_linear_indices(ckpt)
            except Exception:
                pass
            if len(indices) > 0:
                first_w = ckpt[f'encoder.encoders.obs.mlp_head.{indices[0]}.weight']
                in_size = first_w.shape[1]
            else:
                in_size = 1
            obs_mean = torch.zeros(in_size)
            obs_var = torch.ones(in_size)

        # Build encoder dynamically from checkpoint
        encoder = cls._build_encoder_from_ckpt(ckpt, nonlinearity=nonlinearity, jit=jit_encoder)

        # Determine GRU input size from the last linear layer out_features or keep 512 default
        gru_input_size = 512
        lin_indices = cls._find_mlp_linear_indices(ckpt)
        if len(lin_indices) > 0:
            last_idx = lin_indices[-1]
            last_w = ckpt[f'encoder.encoders.obs.mlp_head.{last_idx}.weight']
            gru_input_size = last_w.shape[0]

        # GRU core (hidden size inferred from checkpoint)
        rnn_size = ckpt['core.core.weight_hh_l0'].shape[1]
        core = nn.GRU(input_size=gru_input_size, hidden_size=rnn_size, batch_first=False)
        core.weight_ih_l0.data.copy_(ckpt['core.core.weight_ih_l0'])
        core.weight_hh_l0.data.copy_(ckpt['core.core.weight_hh_l0'])
        core.bias_ih_l0.data.copy_(ckpt['core.core.bias_ih_l0'])
        core.bias_hh_l0.data.copy_(ckpt['core.core.bias_hh_l0'])

        # Identity decoder
        class _Identity(nn.Module):
            def forward(self, x):
                return x

        decoder = _Identity()

        # Distribution head: [mean, logstd]
        dist_in = ckpt['action_parameterization.distribution_linear.weight'].shape[1]
        dist_linear = nn.Linear(dist_in, 6)
        dist_linear.weight.data.copy_(ckpt['action_parameterization.distribution_linear.weight'])
        dist_linear.bias.data.copy_(ckpt['action_parameterization.distribution_linear.bias'])

        policy = cls().to(device)
        policy._init_modules(obs_mean, obs_var, encoder, core, dist_linear, decoder)
        return policy


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="RLPolicyClean one-step inference example")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Sample Factory checkpoint .pth")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--nonlinearity", type=str, default="relu", choices=["relu", "elu", "tanh"], help="Activation to use (must match training)")
    parser.add_argument("--jit_encoder", action="store_true", help="Enable torch.jit.script for encoder MLP")
    parser.add_argument("--normalized", action="store_true", help="Treat provided obs as already normalized")
    args = parser.parse_args()

    # Load policy
    policy = RLPolicyClean.load_from_checkpoint(
        args.ckpt,
        device=args.device,
        nonlinearity=args.nonlinearity,
        jit_encoder=args.jit_encoder,
    ).eval()

    # Prepare a dummy observation (batch=1) matching checkpoint input size
    # If not normalized, obs will be normalized internally using checkpoint stats
    with torch.no_grad():
        in_size = policy.obs_mean.shape[0]
        obs = torch.zeros(1, in_size, device=args.device)
        policy.reset_hidden_state(batch_size=1, device=torch.device(args.device))

        action_logits = policy(obs, normalized=args.normalized)
        print("action_logits shape:", tuple(action_logits.shape))
        print("hidden state shape:", tuple(policy.hxs.shape))
        print("action_logits (first row):", action_logits[0].tolist())

