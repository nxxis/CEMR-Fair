import torch
import torch.nn as nn

def tide_drift_loss(actual, predicted):
    """Mean-squared drift loss used to stabilize continuous predictions.

    Parameters
    ----------
    actual, predicted : Tensor
        Tensors of matching shape containing true and predicted latent
        subsequences used for drift regularization.
    """
    return nn.MSELoss()(actual, predicted)


def evidential_regression_loss(gamma, v, alpha, beta, targets, mask, lambda_reg=0.01):
    """Evidential NLL + regularizer for Normal-Inverse-Gamma outputs.

    This function first masks out padded timesteps (using `mask`), then
    computes the negative log-likelihood of a Normal-Inverse-Gamma
    predictive distribution and an epistemic regularizer that penalizes
    under-confident predictions for large errors.

    Notes
    -----
    - gamma, v, alpha, beta are expected to be tensors shaped
      [Batch*ValidSteps, num_vitals] after masking (the function
      itself performs masking), and `targets` should be aligned.
    - The implementation clamps parameters to maintain numerical
      stability and the statistical constraints (e.g., alpha > 1).
    - `lambda_reg` weights the epistemic regularizer term.
    """
    # Filter down to unpadded timesteps
    gamma = gamma[mask]
    v = v[mask]
    alpha = alpha[mask]
    beta = beta[mask]
    y = targets[mask]

    # 1. Negative Log-Likelihood of Normal-Inverse-Gamma
    # Remap parameters to guarantee valid statistical boundaries
    v = torch.clamp(v, min=1e-6)
    alpha = torch.clamp(alpha, min=1.0 + 1e-6)
    beta = torch.clamp(beta, min=1e-6)

    omg = 2 * beta * (1 + v)
    
    # Standard continuous EDL log-likelihood formulation
    nll = 0.5 * torch.log(torch.pi / v) \
          - alpha * torch.log(omg) \
          + (2 * alpha + 1) * 0.5 * torch.log(v * (y - gamma)**2 + omg) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)

    # 2. Epistemic Uncertainty Regularizer (penalizes high error with low uncertainty)
    error = torch.abs(y - gamma)
    evidence = 2 * v + alpha
    reg = error * evidence

    return torch.mean(nll) + lambda_reg * torch.mean(reg)