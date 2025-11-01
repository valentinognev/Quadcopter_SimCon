import math
import torch
import torch.nn as nn
import sys

# Add sample-factory to path to import utilities
sys.path.insert(0, 'sample-factory')

EPS = 1e-8  # Match Sample Factory's EPS constant from algo.utils.misc

class RLPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # 1) obs normalizer parameters (identity for 3-dim obs)
        self.obs_mean = None #nn.Parameter(torch.zeros(8), requires_grad=False)
        self.obs_var  = None #nn.Parameter(torch.ones(8), requires_grad=False)
        # 2) encoder MLP: 8 → 64 → 64 → 64
        self.encoder = None #nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        # 3) recurrent core: GRU(64 → 512)
        self.core = None #nn.GRU(input_size=64, hidden_size=512, batch_first=True)
        # 4) decoder MLP (may be identity if decoder_mlp_layers is empty)
        self.decoder = None
        # 5) Gaussian head: outputs [mean, logstd] for 3-D action
        self.dist_linear = None #nn.Linear(512, 6)

    def forward(self, obs, hxs, normalized=False):
        """
        obs: Tensor[batch, obs_dim] or already normalized if normalized=True
        hxs: Tensor[batch, 512] or None in SF format
        normalized: if True, obs is already normalized (skip normalization step)
        returns: action_logits[batch, 6], new_hxs[batch, 512]
        """
        # normalize (identity over 3 dims)
        if not normalized:
            x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + EPS)
            # Clip to [-5, 5] like Sample Factory does
            x = torch.clamp(x, -5.0, 5.0)
        else:
            x = obs  # Already normalized
        # encode
        x = self.encoder(x)                        # [batch,64]
        # recurrent
        # GRU expects [seq_len, batch, features] and hxs [num_layers, batch, hidden]
        x = x.unsqueeze(0)                         # [1, batch, 64]
        if hxs is None:
            hxs = torch.zeros(1, x.shape[1], 512, device=x.device)
        elif hxs.dim() == 2:
            hxs = hxs.unsqueeze(0)                  # [1, batch, 512] -> [1, batch, 512]
        x, h_next = self.core(x, hxs)              # x: [1, batch, 512], h_next: [1, batch, 512]
        x = x.squeeze(0)                           # [batch, 512]
        h_next = h_next.squeeze(0)                 # [batch, 512]
        # decoder (if present, otherwise identity)
        if self.decoder is not None:
            x = self.decoder(x)                    # [batch, decoder_out_size]
        # distribution parameters
        params = self.dist_linear(x)               # [batch,6]
        mean, logstd = params.chunk(2, dim=-1)     # each [batch,3]
        # Return full action_logits like SF (mean + logstd concatenated)
        action_logits = torch.cat([mean, logstd], dim=-1)  # [batch,6]
        return action_logits, h_next

    def init_torchData(self, obs_mean, obs_var, encoder, core, dist_linear, decoder=None):
        self.obs_mean = obs_mean
        self.obs_var = obs_var
        self.encoder = encoder
        self.core = core
        self.decoder = decoder
        self.dist_linear = dist_linear
        
    @classmethod
    def load_from_checkpoint(cls, path, device="cpu"):
        """
        Load weights for encoder, core, and dist_linear; skip obs normalizer.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)["model"]
        
        inputSize = ckpt["encoder.encoders.obs.mlp_head.0.weight"].shape[1]
        hiddenSize1 = ckpt["encoder.encoders.obs.mlp_head.2.weight"].shape[0]
        hiddenSize2 = ckpt["encoder.encoders.obs.mlp_head.4.weight"].shape[0]
        
        # Load observation normalizer parameters from checkpoint
        obs_mean_key = "obs_normalizer.running_mean_std.running_mean_std.obs.running_mean"
        obs_var_key = "obs_normalizer.running_mean_std.running_mean_std.obs.running_var"
        
        if obs_mean_key in ckpt and obs_var_key in ckpt:
            obs_mean = nn.Parameter(ckpt[obs_mean_key].float(), requires_grad=False)
            obs_var = nn.Parameter(ckpt[obs_var_key].float(), requires_grad=False)
        else:
            # Fallback to identity normalization if not found
            obs_mean = nn.Parameter(torch.zeros(inputSize), requires_grad=False)
            obs_var  = nn.Parameter(torch.ones(inputSize), requires_grad=False)
        # Use create_mlp from SF's model_utils to match encoder construction exactly
        from sample_factory.model.model_utils import create_mlp, fc_layer, nonlinearity
        
        # Get activation function (should be ELU based on checkpoint)
        # Create a dummy config object with nonlinearity='elu'
        class DummyCfg:
            nonlinearity = 'relu'
        
        dummy_cfg = DummyCfg()
        activation = nonlinearity(dummy_cfg)
        
        # Build MLP using SF's fc_layer and activation
        # Structure per create_mlp: [FC, Act] repeated for each layer size
        # For [64,64,64] this yields indices: 0:FC,1:Act, 2:FC,3:Act, 4:FC,5:Act
        layers = []
        layers.extend([fc_layer(inputSize, hiddenSize1), activation])      # 0, 1
        layers.extend([fc_layer(hiddenSize1, hiddenSize2), activation])    # 2, 3
        layers.extend([fc_layer(hiddenSize2, hiddenSize2), activation])    # 4, 5
        encoder = nn.Sequential(*layers)
        
        rnnSize = 512
        core = nn.GRU(input_size=hiddenSize2, hidden_size=rnnSize, batch_first=False)  # SF doesn't use batch_first
        
        # Load MLP weights BEFORE scripting (can't modify weights after scripting)
        encoder[0].weight.data.copy_(ckpt["encoder.encoders.obs.mlp_head.0.weight"])
        encoder[0].bias.data  .copy_(ckpt["encoder.encoders.obs.mlp_head.0.bias"])
        encoder[2].weight.data.copy_(ckpt["encoder.encoders.obs.mlp_head.2.weight"])
        encoder[2].bias.data  .copy_(ckpt["encoder.encoders.obs.mlp_head.2.bias"])
        encoder[4].weight.data.copy_(ckpt["encoder.encoders.obs.mlp_head.4.weight"])
        encoder[4].bias.data  .copy_(ckpt["encoder.encoders.obs.mlp_head.4.bias"])
        
        # JIT only the encoder (first stage) to match SF's behavior
        # This allows us to compare encoder outputs stage-by-stage
        # Do NOT JIT MLP encoder to match Sample Factory MlpEncoder (SF keeps it unscripted)
        # load GRU weights
        core.weight_ih_l0.data.copy_(ckpt["core.core.weight_ih_l0"])
        core.weight_hh_l0.data.copy_(ckpt["core.core.weight_hh_l0"])
        core.bias_ih_l0  .data.copy_(ckpt["core.core.bias_ih_l0"])
        core.bias_hh_l0  .data.copy_(ckpt["core.core.bias_hh_l0"])
        # Create decoder (identity in this case since decoder_mlp_layers is empty)
        class IdentityDecoder(nn.Module):
            def forward(self, x):
                return x
        decoder = IdentityDecoder()
        
        # load distribution head
        # dist_linear input size matches the decoder output (which is 512 for this model)
        dist_linear_input_size = ckpt["action_parameterization.distribution_linear.weight"].shape[1]
        dist_linear = nn.Linear(dist_linear_input_size, 6)
        dist_linear.weight.data.copy_(ckpt["action_parameterization.distribution_linear.weight"])
        dist_linear.bias.data  .copy_(ckpt["action_parameterization.distribution_linear.bias"])
        
        net = cls().to(device)
        net.init_torchData(obs_mean, obs_var, encoder, core, dist_linear, decoder)
        return net
