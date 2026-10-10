"""
MLP-based Transmission Energy Predictor for LoRaWAN Clustering
Paper: Prediction-Guided Energy-Aware Clustering for Dense LoRaWAN Networks
Author: Cristofe Rocha et al.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Neural Network Architecture (Section III-B)
# -----------------------------------------------------------------------------
class MLPEnergyPredictor(nn.Module):
    """
    Fully-connected 5 -> 64 -> 32 -> 1 MLP with ReLU activations and linear output.
    Input features: [SF, BW, RSSI, SNR, ToA]
    Output: Estimated transmission energy in mJ
    """
    def __init__(self, input_dim=5, hidden1=64, hidden2=32, output_dim=1):
        super(MLPEnergyPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# -----------------------------------------------------------------------------
# 2. Training and Evaluation Pipeline
# -----------------------------------------------------------------------------
def train_and_evaluate(data_path="recent_logs/links.csv", output_dir="output", epochs=30, batch_size=64, lr=0.005, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    print(f"[*] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    features = ['sf', 'bw', 'rssi', 'snr', 'toa']
    df.dropna(subset=features + ['energy_j'], inplace=True)

    X = df[features].values.astype(np.float32)
    # Convert transmission energy from Joules to millijoules (mJ)
    y = (df['energy_j'].values * 1000.0).astype(np.float32).reshape(-1, 1)

    # 2. 70% Train, 15% Validation, 15% Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed
    )

    print(f"    - Training set:   {len(X_train)} samples (70%)")
    print(f"    - Validation set: {len(X_val)} samples (15%)")
    print(f"    - Test set:       {len(X_test)} samples (15%)")

    # 3. Z-score Standardization (strictly fitted on training set)
    mean_train = X_train.mean(axis=0)
    std_train = X_train.std(axis=0) + 1e-8

    X_train_norm = (X_train - mean_train) / std_train
    X_val_norm = (X_val - mean_train) / std_train
    X_test_norm = (X_test - mean_train) / std_train

    np.savez(output_dir / "scaler_params.npz", mean=mean_train, std=std_train)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_norm), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True
    )
    X_val_t = torch.tensor(X_val_norm).to(device)
    y_val_t = torch.tensor(y_val).to(device)

    # 4. Model, Loss, Optimizer and Scheduler
    model = MLPEnergyPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses, val_losses = [], []

    # 5. Training loop
    print(f"\n[*] Training MLP for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(bx)

        scheduler.step()
        epoch_train_loss = running_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            epoch_val_loss = criterion(val_preds, y_val_t).item()

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch [{epoch:02d}/{epochs:02d}] | Train MSE: {epoch_train_loss:.6f} mJ² | Val MSE: {epoch_val_loss:.6f} mJ²")

    # Save weights
    torch.save(model.state_dict(), output_dir / "mlp_model.pt")
    print(f"\n[✓] Trained model weights saved to: {output_dir / 'mlp_model.pt'}")

    # 6. Test evaluation
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_norm).to(device)
        y_test_pred = model(X_test_t).cpu().numpy()

    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    errors = (y_test - y_test_pred).flatten()

    pct_01 = (np.abs(errors) < 0.01).mean() * 100.0
    pct_05 = (np.abs(errors) < 0.05).mean() * 100.0

    print("\n" + "=" * 55)
    print("TEST SET EVALUATION METRICS (Table II in manuscript)")
    print("=" * 55)
    print(f"MSE:              {mse:.2e} mJ² ({mse:.6f})")
    print(f"RMSE:             {rmse:.4f} mJ")
    print(f"MAE:              {mae:.4f} mJ")
    print(f"R²:               {r2:.5f} (>0.999)")
    print(f"|ε| < 0.01 mJ:    {pct_01:.2f}%")
    print(f"|ε| < 0.05 mJ:    {pct_05:.2f}%")
    print("=" * 55)

    # 7. Generate convergence and residual plots
    generate_plots(train_losses, val_losses, errors, output_dir)

def generate_plots(train_losses, val_losses, errors, output_dir):
    epochs = len(train_losses)
    
    # Loss plot
    plt.figure(figsize=(5.5, 4.0))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='#1f77b4', lw=2)
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='#ff7f0e', lw=2, linestyle='--')
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel(r'MSE Loss ($\mathrm{mJ}^2$)', fontsize=11)
    plt.title('MLP Training Convergence (30 Epochs)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "mlp_loss.png", dpi=300)
    plt.close()

    # Residuals plot
    plt.figure(figsize=(5.5, 4.0))
    plt.hist(errors, bins=25, color='#2ca02c', edgecolor='black', alpha=0.7)
    plt.xlabel(r'Residual Error $y - \widehat{y}$ (mJ)', fontsize=11)
    plt.ylabel('Frequency (Sample Count)', fontsize=11)
    plt.title('Test-Set Residual Distribution (N=883)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "residuos.png", dpi=300)
    plt.close()
    print(f"[✓] Plots generated in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP Energy Predictor for LoRaWAN")
    parser.add_argument("--data", type=str, default="recent_logs/links.csv", help="Path to links.csv dataset")
    parser.add_argument("--out", type=str, default="output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    args = parser.parse_args()

    train_and_evaluate(args.data, args.out, epochs=args.epochs, lr=args.lr)
