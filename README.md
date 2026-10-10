# RAMO: Prediction-Guided Energy-Aware Clustering for Dense LoRaWAN Networks

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official code and dataset repository for the paper:  
**"Prediction-Guided Energy-Aware Clustering for Dense LoRaWAN Networks"**  
*Cristofe Rocha et al.*

---

## 📌 Overview

Clustering mitigates channel contention in dense LoRaWAN deployments, but conventional algorithms select Cluster Heads (CHs) based solely on geometric location or *current* residual energy. Because transmission costs vary substantially with physical-layer link conditions (Spreading Factor, SNR, RSSI, and Time-on-Air), nodes with high initial energy may suffer severe degradation if elected under poor channel states.

**RAMO (Residual-energy and Area-density Multi-Objective)** introduces an end-to-end **prediction-to-decision framework**:
1. **Supervised Energy Estimation:** A Multi-Layer Perceptron (MLP) learns non-linear radio transmission costs directly from physical-layer communication features ($\mathrm{SF}, \mathrm{BW}, \mathrm{RSSI}, \mathrm{SNR}, \mathrm{ToA}$).
2. **Projected Residual Energy:** The learned estimate projects future battery states ($E_{i,t+1}^{\mathrm{pred}} = E_{i,t} - \widehat{E}_{i,t+1}$) before CH election.
3. **Density-Aware Spatial Clustering:** Kernel Density Estimation (KDE) and K-Means partition the network into optimal topological clusters.
4. **Multi-Objective CH Selection:** Balances weak centroid proximity, projected residual energy, node density, and gateway distance.

$$\mathbf{x}_{i,t} \longrightarrow \widehat{E}_{i,t+1} \longrightarrow E_{i,t+1}^{\mathrm{pred}} \longrightarrow \mathrm{RAMO\ Decision}$$

---

## 📂 Repository Structure

```text
├── data/ (or recent_logs/)
│   ├── links.csv                 # 5,883 transmission instances logged from LoRaSimPlus
│   ├── node.txt                  # Node deployment coordinates and initial parameters
│   └── mlp_predicted_energy.csv  # Pre-computed MLP energy predictions per node
├── output/
│   ├── mlp_model.pt              # Trained PyTorch MLP weights
│   ├── scaler_params.npz         # Z-score standardization parameters (causal)
│   ├── mlp_loss.png              # Training and validation loss curves (30 epochs)
│   └── residuos.png              # Test-set error distribution histogram (N = 883)
├── train_mlp.py                  # Standalone MLP training, validation, and evaluation script
├── ramo.py                       # RAMO multi-objective clustering and CH selection
├── simulate_lifetime.py          # Long-term network survival simulation (1,152 rounds)
├── calculate_k_opt.py            # Optimal cluster count determination via KDE
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/cristofelr/ramo.git
cd ramo
pip install -r requirements.txt
```

### 2. Train the MLP Energy Predictor

Train the PyTorch MLP on the 5,883 transmission samples across 30 epochs:

```bash
python train_mlp.py --data recent_logs/links.csv --out output --epochs 30
```

**Output Metrics on Held-Out Test Set ($N = 883$):**

| Metric | Empirical Result |
|---|---|
| **Mean Squared Error (MSE)** | $7.86 \times 10^{-5}\text{ mJ}^2$ |
| **Root Mean Squared Error (RMSE)** | $0.0089\text{ mJ}$ |
| **Mean Absolute Error (MAE)** | $0.0037\text{ mJ}$ |
| **Coefficient of Determination ($R^2$)** | $> 0.999$ ($0.99997$) |
| **Samples with $\|\epsilon\| < 0.01\text{ mJ}$** | **91.05%** |
| **Samples with $\|\epsilon\| < 0.05\text{ mJ}$** | **99.32%** |

### 3. Run RAMO Clustering & CH Selection

Execute the multi-objective cluster head election:

```bash
# Using predictive projected residual energy (Predictive RAMO)
python ramo.py --battery predictive

# Using current residual energy (Reactive RAMO baseline)
python ramo.py --battery residual
```

### 4. Run Network Survival Simulation

Simulate full battery exhaustion over 1,152 rounds:

```bash
python simulate_lifetime.py
```

---

## 📊 Summary of Paper Results

Evaluated on a 500-node Smart City scenario in LoRaSimPlus over a 1-hour window ($T = 3{,}600\text{ s}$):

| Configuration | Transmitted | Received | PDR (%) | Goodput (bps) | EBI ($\uparrow$) | CV ($\downarrow$) |
|---|---|---|---|---|---|---|
| **Direct LoRaWAN** | 5,928 | 4,355 | 73.46% | 193.56 | — | — |
| **Conventional K-Means** | 6,240 | 5,210 | 83.49% | 231.56 | 0.157 | 0.064 |
| **Reactive RAMO** | 5,886 | 5,044 | 85.69% | 224.18 | 0.039 | 0.133 |
| **Predictive RAMO (Ours)** | **5,883** | **5,403** | **91.84%** | **240.13** | **0.179** | **0.107** |

*Predictive RAMO achieves a **42.9% reduction in packet losses**, **+6.15 p.p. in PDR**, a **+355% increase in Energy Balancing Index (EBI)**, and a **26.1% reduction in transmission energy per delivered packet**.*

---

## 📜 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{rocha2026ramo,
  author    = {Cristofe Rocha and others},
  title     = {Prediction-Guided Energy-Aware Clustering for Dense {LoRaWAN} Networks},
  booktitle = {Proceedings of the IEEE Conference},
  year      = {2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
