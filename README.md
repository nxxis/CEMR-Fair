# CEMR-Fair: Continuous-Time Evidential Multimodal Representation Learning for Fairness

This repository contains the official PyTorch implementation for **CEMR-Fair**, a framework designed to model irregularly sampled, continuous-time clinical trajectories while mathematically guaranteeing demographic parity through evidential uncertainty calibration.

This codebase was developed and evaluated using the MIMIC-IV clinical dataset.

## 📌 Features
* **Continuous-Time Modeling:** Utilizes Neural Ordinary Differential Equations (ODEs) to handle irregular sampling intervals in ICU telemetry without rigid imputation.
* **Evidential Deep Learning:** Projects latent states into a Normal-Inverse-Gamma distribution to quantify Epistemic (model) Uncertainty.
* **Minimax Fairness:** Employs an adversarial discriminator to ensure latent representations and uncertainty bounds are statistically blind to protected demographic attributes (Age and Gender).

## ⚙️ Environment Setup
To ensure strict reproducibility, we have locked all dependencies to the exact versions used during our evaluation. 

Requires Python 3.10+ and a CUDA-enabled GPU (NVIDIA T4 or higher recommended).

```bash
# Clone the repository
git clone [https://github.com/nxxis/CEMR-Fair](https://github.com/nxxis/CEMR-Fair)
cd CEMR-Fair

# Install strict dependencies
pip install -r requirements.txt