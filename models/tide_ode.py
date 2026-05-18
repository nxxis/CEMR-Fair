"""Model definitions for the continuous-time TIDE / CEMR ODEs.

This module provides a small ODE function and a CEMR evidential
wrapper that maps continuous latent trajectories to evidential
distribution parameters per vital sign.

Shapes and conventions
----------------------
- Inputs to `forward` are `x_seq` with shape [Seq, Batch, feat]
    and `t_eval` a 1D tensor of times to evaluate the ODE on.
- Outputs: `full_trajectory` is [Seq, Batch, feat]; `z_actual` and
    `z_pred` are sequence-aligned tensors used for drift regularization.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """Simple MLP used as the vector field f(y) for the ODE solver.

    Args
    ----
    latent_dim: int
        Dimensionality of the latent state y.
    """
    def __init__(self, latent_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim)
        )
    def forward(self, t, y):
        # `odeint` calls this with signature (t, y)
        return self.net(y)

class CEMREvidentialODE(nn.Module):
    """Evidential Neural ODE that produces trajectories + evidential params.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent state used internally by the ODE.
    num_vitals : int
        Number of parallel vital sign outputs (each has 4 evidential params).
    num_confounders : int
        Output dimension of the confounder discriminator.
    """
    def __init__(self, latent_dim=16, num_vitals=4, num_confounders=2):
        super(CEMREvidentialODE, self).__init__()
        self.dynamics = ODEFunc(latent_dim)
        self.num_vitals = num_vitals
        
        # Maps the dense latent trajectory state to Evidential Distribution Parameters
        # 4 parameters (gamma, v, alpha, beta) per vital sign
        self.evidential_head = nn.Linear(latent_dim, num_vitals * 4)
        
        # Confounder Discriminator for algorithmic fairness
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_confounders)
        )

    def forward(self, x_seq, t_eval):
        """Evaluate the ODE and map outputs to evidential params.

        Inputs
        ------
        x_seq : Tensor
            Input sequence with shape [Seq, Batch, feat]. The ODE initial
            condition is taken as `x_seq[0]`.
        t_eval : Tensor
            1D times at which to evaluate the ODE.

        Returns
        -------
        full_trajectory : Tensor
            Shape [Seq, Batch, feat] — continuous trajectory evaluated at t_eval.
        z_actual : Tensor
            The input subsequence used for drift regularization (x_seq[1:]).
        z_pred : Tensor
            The matching predicted subsequence from the ODE (full_trajectory[1:]).
        (gamma, v, alpha, beta) : tuple of Tensors
            Evidential parameters reshaped to [Seq, Batch, num_vitals].
        """
        # Solve continuous trajectory over the irregular timeline
        full_trajectory = odeint(self.dynamics, x_seq[0], t_eval, method="rk4")
        
        # Extract sequence steps for continuous drift regularization
        z_actual = x_seq[1:] 
        z_pred = full_trajectory[1:]
        
        # Pass the trajectory through the evidential parameter mapping layer
        seq_len, batch_size, _ = full_trajectory.shape
        raw_params = self.evidential_head(full_trajectory) # Shape: [Seq, Batch, num_vitals*4]
        
        # Reshape to [Seq, Batch, Num_Vitals, 4_Parameters]
        raw_params = raw_params.view(seq_len, batch_size, self.num_vitals, 4)
        
        # Apply structural activation constraints to guarantee valid statistical bounds
        # `gamma` can be any real value (mean proxy). The other params are
        # constrained via exponentials and offsets to ensure positivity / alpha>1.
        gamma = raw_params[..., 0]                    # Real-valued mean proxy
        v = torch.exp(raw_params[..., 1]) + 1e-6       # Degrees of freedom (Positive)
        alpha = torch.exp(raw_params[..., 2]) + 1.0    # Shape parameter (Alpha > 1)
        beta = torch.exp(raw_params[..., 3]) + 1e-6    # Scale parameter (Positive)
        
        return full_trajectory, z_actual, z_pred, (gamma, v, alpha, beta)