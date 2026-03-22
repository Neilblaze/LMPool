import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan


"""
- A Mamba model consists of multiple layers, which are ResidualBlocks.

- A ResidualBlock is composed of a MambaBlock, a normalization layer, and a residual connection:
  ResidualBlock(x) = mamba(norm(x)) + x

- This leads to the MambaBlock: its input x is (B, L, D) and output y is also (B, L, D)
  (B=batch size, L=sequence length, D=model dimension).
  First, x is expanded to (B, L, 2*ED) (where E is typically 2) and split into x and z, each (B, L, ED).
  Then a short 1D convolution is applied to x, followed by a SiLU activation, then the SSM.
  The result is then multiplied by silu(z).

"""


@dataclass
class MambaConfig:
    """
    Configuration class for the Mamba model, storing various hyperparameters.

    Attributes:
        d_model (int): Model dimension D.
        n_layers (int): Number of layers in the model.
        dt_rank (Union[int, str], optional): Rank of the discrete-time matrix. Set to 'auto' to compute automatically. Defaults to 'auto'.
        d_state (int, optional): State dimension N (referred to as N in the paper and comments). Defaults to 16.
        expand_factor (int, optional): Expansion factor E (referred to as E in the paper and comments). Defaults to 2.
        d_conv (int, optional): Convolution layer dimension. Defaults to 4.

        dt_min (float, optional): Minimum value for the discrete time step. Defaults to 0.001.
        dt_max (float, optional): Maximum value for the discrete time step. Defaults to 0.1.
        dt_init (str, optional): Initialization method for the discrete time step, either 'random' or 'constant'. Defaults to 'random'.
        dt_scale (float, optional): Scaling factor for the discrete time step. Defaults to 1.0.
        dt_init_floor (float, optional): Lower bound for the discrete time step initialization. Defaults to 1e-4.

        rms_norm_eps (float, optional): Epsilon parameter for RMS normalization. Defaults to 1e-5.
        base_std (float, optional): Base standard deviation used for parameter initialization. Defaults to 0.02.

        bias (bool, optional): Whether to use bias. Defaults to False.
        conv_bias (bool, optional): Whether the convolution layer uses bias. Defaults to True.
        inner_layernorms (bool, optional): Whether to apply layer normalization to internal activations. Defaults to False.

        mup (bool, optional): Whether to use muP (model parallelism). Defaults to False.
        mup_base_width (float, optional): Base width for muP. Defaults to 128.

        pscan (bool, optional): Whether to use parallel scan mode during training. If False, uses sequential mode. Defaults to True.
        use_cuda (bool, optional): Whether to use the official CUDA implementation during training (not compatible with (b)float16). Defaults to False.
    """
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        # Compute the inner dimension D_inner = E * D.
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        # Compute dt_rank automatically if set to 'auto'.
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP setup.
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class Mamba(nn.Module):
    """
    Mamba model class implementing the Mamba architecture.

    The model is composed of multiple ResidualBlocks, each containing a MambaBlock,
    a normalization layer, and a residual connection.

    Args:
        config (MambaConfig): Configuration for the Mamba model, containing various hyperparameters.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()

        # Store the config.
        self.config = config

        # Build the model layers; each layer is a ResidualBlock.
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        """
        Forward pass.

        Processes the input tensor x through multiple ResidualBlock layers.

        Args:
            x (Tensor): Input tensor of shape (B, L, D).

        Returns:
            Tensor: Output tensor of shape (B, L, D).
        """
        # x has shape (B, L, D)

        # Pass through each layer.
        for layer in self.layers:
            x = layer(x)

        return x

    def step(self, x, caches):
        """
        Single-step forward pass for autoregressive inference.

        Args:
            x (Tensor): Input tensor of shape (B, L, D).
            caches (List[dict]): Per-layer cache list; each cache is a dict holding historical state.

        Returns:
            Tuple[Tensor, List[dict]]: Processed output tensor and updated cache list.
        """
        # x has shape (B, L, D)
        # caches is a list of per-layer caches with shape (h, inputs)

        # Process each layer and update its cache.
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    """
    Residual block used to build the Mamba model.

    This module is composed of a MambaBlock, a normalization layer, and a residual connection.
    Specifically, the output of a ResidualBlock is the sum of the MambaBlock output and the input:
        ResidualBlock(x) = MambaBlock(norm(x)) + x

    Args:
        config (MambaConfig): Configuration for the Mamba model, containing various hyperparameters.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()

        # Initialize MambaBlock as the mixer.
        self.mixer = MambaBlock(config)
        # Initialize RMS normalization with dimension d_model, epsilon rms_norm_eps, and mup flag.
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        """
        Forward pass.

        Normalizes the input tensor x, passes it through MambaBlock, and adds the original input
        to implement the residual connection.

        Args:
            x (Tensor): Input tensor of shape (B, L, D).

        Returns:
            Tensor: Output tensor of shape (B, L, D).
        """
        # Normalize the input tensor x.
        # x has shape (B, L, D)

        # Pass the normalized tensor through MambaBlock.
        # Output shape: (B, L, D)
        output = self.mixer(self.norm(x)) + x
        return output

    def step(self, x, cache):
        """
        Single-step forward pass for autoregressive inference.

        Args:
            x (Tensor): Input tensor of shape (B, D).
            cache (dict): Cache for this layer, containing historical state.

        Returns:
            Tuple[Tensor, dict]: Processed output tensor and updated cache.
        """
        # Normalize the input tensor x.
        # x has shape (B, D)
        # cache has shape (h, inputs)
        # h has shape (B, ED, N)
        # inputs has shape (B, ED, d_conv-1)

        # Pass the normalized tensor through MambaBlock's single-step forward pass, and update the cache.
        # Output shape: (B, D)
        # cache shape remains unchanged.
        output, cache = self.mixer.step(self.norm(x), cache)

        # Add the output to the original input x to implement the residual connection.
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    """
    Core building block of the Mamba model.

    MambaBlock is the central component of Mamba, responsible for processing input features
    and performing state-space modeling. It achieves complex feature transformations through
    linear projections, 1D convolution, a state-space model (SSM), and residual connections.

    Args:
        config (MambaConfig): Configuration for the Mamba model, containing various hyperparameters.
    """
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # Projects input from D to 2*ED (two branches).
        # ED is the expanded dimension, typically a multiple of D, used to widen the feature representation.
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        # 1D depthwise convolution applied over the time dimension.
        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                              kernel_size=config.d_conv, bias=config.conv_bias,
                              groups=config.d_inner,
                              padding=config.d_conv - 1)

        # Projects the input tensor to input-dependent delta, B, C.
        # delta: the step size used to update the state-space model.
        # B, C: parameters of the state-space model.
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # Projects delta from dt_rank to d_inner.
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        # Compute the initialization standard deviation for dt.
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            # Initialize dt_proj weights with a constant value.
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            # Initialize dt_proj weights with a uniform random distribution.
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            # Raise an error if the initialization method is not implemented.
            raise NotImplementedError

        # Initialize the dt bias.
        # Compute initial dt values using an exponential function and a random number generator.
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        # Compute the softplus inverse to initialize the dt bias.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization.
        # Generate the state-space model parameter A.
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        # Store A in log space to enforce A < 0, ensuring numerical stability.
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        # Exclude A_log from weight decay.
        self.A_log._no_weight_decay = True

        # Initialize the state-space model parameter D.
        # Initialize D to all ones.
        self.D = nn.Parameter(torch.ones(config.d_inner))
        # Exclude D from weight decay.
        self.D._no_weight_decay = True

        # Projects block output from ED back to D.
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # Used in Jamba.
        if self.config.inner_layernorms:
            # If inner layer normalization is enabled, define RMS normalization layers.
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
        else:
            # Otherwise, set normalization layers to None.
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            # If using CUDA, attempt to import the CUDA selective scan implementation.
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                # If import fails, print an error and fall back to the Python implementation.
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        """
        Applies layer normalization to dt, B, and C.

        Args:
            dt (Tensor): Delta tensor.
            B (Tensor): B tensor.
            C (Tensor): C tensor.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Normalized dt, B, and C.
        """
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        """
        Forward pass.

        Processes the input tensor x through linear projection, 1D convolution, SiLU activation,
        the state-space model (SSM), and output projection.

        Args:
            x (Tensor): Input tensor of shape (B, L, D).

        Returns:
            Tensor: Output tensor of shape (B, L, D).
        """
        # Get batch size B, sequence length L, and other dimensions from the input tensor.
        _, L, _ = x.shape

        # Project the input tensor to 2*ED (two branches).
        xz = self.in_proj(x) # (B, L, 2*ED)

        # Split the projected tensor into branches x and z.
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch processing:
        # Transpose x from (B, L, ED) to (B, ED, L).
        x = x.transpose(1, 2) # (B, ED, L)

        # Apply 1D convolution to x with kernel size `config.d_conv` and causal padding.
        # Then slice the first L elements to preserve the sequence length.
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter

        # Transpose x from (B, ED, L) back to (B, L, ED).
        x = x.transpose(1, 2) # (B, L, ED)

        # Apply SiLU activation to x.
        x = F.silu(x)

        # Pass x and z through the state-space model (SSM) to get output y.
        y = self.ssm(x, z)

        # CUDA path.
        if self.config.use_cuda:
            # Project y to the final output via the output projection layer.
            output = self.out_proj(y) # (B, L, D)
            return output # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

        # z branch processing.
        # Apply SiLU activation to z.
        z = F.silu(z) # (B, L, D)

        # Multiply y and z to produce the output.
        output = y * z # (B, L, ED)

        # Project the output to the final dimension via the output projection layer.
        output = self.out_proj(output) # (B, L, D)

        return output

    def ssm(self, x, z):
        """
        State-space model (SSM) method.

        Applies state-space modeling to input features x and auxiliary vector z.

        Args:
            x (Tensor): Input feature tensor of shape (B, L, ED).
            z (Tensor): Auxiliary vector tensor of shape (B, L, ED).

        Returns:
            Tensor: SSM output of shape (B, L, ED).
        """
        # Compute parameters A and D.
        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        # Project input x to delta, B, C.
        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)

        # Split deltaBC into delta, B, C.
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)

        # Apply layer normalization to delta, B, C.
        delta, B, C = self._apply_layernorms(delta, B, C)

        # Project delta from dt_rank to d_inner.
        delta = self.dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)

        # Choose the selective scan function based on config.
        if self.config.use_cuda:
            # Transpose tensors to match the expected layout for the CUDA implementation.
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused together.
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
            # Transpose output back to (B, L, ED).
            y = y.transpose(1, 2) # (B, L, ED)

        else:
            # Transpose delta and apply softplus with bias.
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias) # (B, L, ED)

            # Choose between parallel scan and sequential scan based on config.
            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D) # (B, L, ED)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D) # (B, L, ED)

        return y

    def selective_scan(self, x, delta, A, B, C, D):
        """
        Selective scan method (parallel version).

        Applies the selective scan operation to input tensor x, combining step size delta
        and parameters A, B, C, D to produce the output.

        Args:
            x (Tensor): Input tensor of shape (B, L, ED).
            delta (Tensor): Step size tensor of shape (B, L, ED).
            A (Tensor): Parameter tensor of shape (ED, N).
            B (Tensor): Parameter tensor of shape (B, L, N).
            C (Tensor): Parameter tensor of shape (B, L, N).
            D (Tensor): Parameter tensor of shape (ED,).

        Returns:
            Tensor: Selective scan output of shape (B, L, ED).
        """
        # Compute deltaA of shape (B, L, ED, N).
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        # Compute deltaB of shape (B, L, ED, N).
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        # Compute BX of shape (B, L, ED, N).
        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        # Call the parallel scan function pscan to get hs of shape (B, L, ED, N).
        hs = pscan(deltaA, BX)

        # Compute output y of shape (B, L, ED, 1).
        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        # Add D * x to y to produce the final output of shape (B, L, ED).
        y = y + D * x

        return y

    def selective_scan_seq(self, x, delta, A, B, C, D):
        """
        Selective scan method (sequential version).

        Applies the selective scan operation to input tensor x, combining step size delta
        and parameters A, B, C, D to produce the output.

        Args:
            x (Tensor): Input tensor of shape (B, L, ED).
            delta (Tensor): Step size tensor of shape (B, L, ED).
            A (Tensor): Parameter tensor of shape (ED, N).
            B (Tensor): Parameter tensor of shape (B, L, N).
            C (Tensor): Parameter tensor of shape (B, L, N).
            D (Tensor): Parameter tensor of shape (ED,).

        Returns:
            Tensor: Selective scan output of shape (B, L, ED).
        """
        # Get batch size B and sequence length L from the input tensor.
        _, L, _ = x.shape

        # Compute deltaA of shape (B, L, ED, N).
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        # Compute deltaB of shape (B, L, ED, N).
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        # Compute BX of shape (B, L, ED, N).
        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        # Initialize h of shape (B, ED, N).
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        # Initialize hs list to accumulate intermediate states.
        hs = []

        # Sequential scan.
        for t in range(0, L):
            # Update h of shape (B, ED, N).
            h = deltaA[:, t] * h + BX[:, t]
            # Append h to the hs list.
            hs.append(h)

        # Stack the hs list into a tensor of shape (B, L, ED, N).
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        # Compute output y of shape (B, L, ED, 1).
        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        # Add D * x to y to produce the final output of shape (B, L, ED).
        y = y + D * x

        return y

    # -------------------------- inference -------------------------- #
    """
    On autoregressive inference

    The advantage of Mamba is that inference time is independent of sequence length.
    We only need to keep two things per layer in the cache:
    - The hidden state h (shape (B, ED, N)), as in RNN-style inference.
    - The last d_conv-1 inputs to the layer, so that the 1D convolution (which operates over time)
      can be computed.
      (d_conv is fixed, so the cache does not grow as the sequence is generated.)
      (d_conv is also typically small, e.g. 4, so we only need to "remember" the last 3 inputs.)

    Concretely, these two quantities are stored in a cache tuple named h and inputs.
    h has shape (B, ED, N), inputs has shape (B, ED, d_conv-1).
    MambaBlock.step() receives this cache and, in addition to its output, returns the updated cache
    for the next call.

    The cache object is initialized as: (None, torch.zeros()).
    When h is None, the selective scan function detects this and starts from h=0.
    torch.zeros() is fine (it is equivalent to providing only the input, since conv1d is padded).

    Since we need one such cache variable per layer, we store a cache object that is simply
    a list of per-layer cache objects.
    """

    def step(self, x, cache):
        """
        Single-step forward pass for autoregressive inference.

        Args:
            x (Tensor): Input tensor of shape (B, D).
            cache (Tuple[Optional[Tensor], Tensor]): Current layer cache containing
                historical hidden state and input buffer.

        Returns:
            Tuple[Tensor, Tuple[Optional[Tensor], Tensor]]: Processed output tensor and updated cache.
        """
        # Unpack hidden state h and input buffer inputs from cache.
        h, inputs = cache  # h: (B, ED, N), inputs: (B, ED, d_conv-1)

        # Project input x to 2*ED (two branches).
        xz = self.in_proj(x) # (B, 2*ED)

        # Split the projected tensor into branches x and z.
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch processing.
        # Expand x to enable concatenation with the input buffer inputs.
        x_cache = x.unsqueeze(2)  # (B, ED, 1)

        # Concatenate input buffer inputs and x_cache along the time dimension.
        # Pass through the 1D convolution layer, then take the last time step.
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        # Apply SiLU activation to x.
        x = F.silu(x)  # (B, ED)

        # Process x and h through the SSM single-step forward pass to get output y and new hidden state h.
        y, h = self.ssm_step(x, h)  # y: (B, ED), h: (B, ED, N)

        # z branch processing.
        # Apply SiLU activation to z.
        z = F.silu(z)  # (B, ED)

        # Multiply y and z to produce the output.
        output = y * z  # (B, ED)

        # Project the output to the final dimension via the output projection layer.
        output = self.out_proj(output) # (B, D)

        # Slide the input buffer left by one step and append the new x_cache.
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)

        # Pack the new hidden state h and updated input buffer into the cache.
        cache = (h, inputs)

        # Return output and updated cache.
        return output, cache

    def ssm_step(self, x, h):
        """
        Single-step forward pass of the state-space model (SSM).

        Applies state-space modeling to input feature x and hidden state h,
        producing output y and the new hidden state h.

        Args:
            x (Tensor): Input feature tensor of shape (B, ED).
            h (Optional[Tensor]): Hidden state tensor of shape (B, ED, N). Initialized to zero if None.

        Returns:
            Tuple[Tensor, Tensor]: Output tensor y and new hidden state h,
                with shapes (B, ED) and (B, ED, N) respectively.
        """
        # Compute parameters A and D.
        A = -torch.exp(self.A_log.float()) # A: (ED, N)
        D = self.D.float() # D: (ED,)

        # Project input x to delta, B, C.
        deltaBC = self.x_proj(x) # deltaBC: (B, dt_rank + 2 * N)

        # Split deltaBC into delta, B, C.
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # delta: (B, dt_rank), B: (B, N), C: (B, N)

        # Apply layer normalization to delta, B, C.
        delta, B, C = self._apply_layernorms(delta, B, C) # delta: (B, dt_rank), B: (B, N), C: (B, N)

        # Project delta from dt_rank to d_inner.
        delta = F.softplus(self.dt_proj(delta)) # delta: (B, ED)

        # Compute deltaA of shape (B, ED, N).
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # deltaA: (B, ED, N)
        # Compute deltaB of shape (B, ED, N).
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # deltaB: (B, ED, N)

        # Compute BX of shape (B, ED, N).
        BX = deltaB * (x.unsqueeze(-1)) # BX: (B, ED, N)

        # If hidden state h is None, initialize it to zeros.
        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # h: (B, ED, N)

        # Update hidden state h of shape (B, ED, N).
        h = deltaA * h + BX  # h: (B, ED, N)

        # Compute output y of shape (B, ED, 1).
        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        # Add D * x to y to produce the final output y of shape (B, ED).
        y = y + D * x  # y: (B, ED)

        return y, h


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).

    Normalizes the input tensor by its root mean square to stabilize training and accelerate convergence.

    Args:
        d_model (int): Dimension of the input features.
        eps (float, optional): Small constant to prevent division by zero. Defaults to 1e-5.
        use_mup (bool, optional): Whether to use muP (model parallelism). Defaults to False.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        # Whether to use muP.
        self.use_mup = use_mup
        # Small constant to prevent division by zero.
        self.eps = eps

        # If not using muP, define a learnable weight parameter to scale the normalized output.
        if not use_mup:
            # Learnable scale parameter.
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        Forward pass.

        Applies RMS normalization to the input tensor, with optional scaling depending on muP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Normalized output tensor.
        """
        # Compute the RMS-normalized output.
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # If not using muP, scale the output.
        if not self.use_mup:
            # Apply the learnable weight parameter.
            return output * self.weight
        else:
            # Return the normalized output directly.
            return output
