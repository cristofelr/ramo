import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

torch.manual_seed(42)
np.random.seed(42)

# 1. Load dataset
df = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/links.csv')
df.dropna(subset=['sf', 'bw', 'rssi', 'snr', 'toa', 'energy_j'], inplace=True)

features = ['sf', 'bw', 'rssi', 'snr', 'toa']
X = df[features].values.astype(np.float32)
y = (df['energy_j'].values * 1000.0).astype(np.float32).reshape(-1, 1)

# 2. Partition 70/15/15
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# 3. Standardize
mean_train = X_train.mean(axis=0)
std_train = X_train.std(axis=0) + 1e-8

X_train_norm = (X_train - mean_train) / std_train
X_val_norm = (X_val - mean_train) / std_train
X_test_norm = (X_test - mean_train) / std_train

# 4. Define MLP architecture (5 -> 64 -> 32 -> 1)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))

model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train_norm), torch.tensor(y_train)), batch_size=64, shuffle=True)

X_val_t = torch.tensor(X_val_norm)
y_val_t = torch.tensor(y_val)

train_losses, val_losses = [], []

# 5. Train over 30 epochs
for epoch in range(30):
    model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(bx)
    scheduler.step()
    train_loss = running_loss / len(X_train)
    
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
        
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# 6. Evaluate on test set (N = 883)
model.eval()
with torch.no_grad():
    y_test_pred = model(torch.tensor(X_test_norm)).numpy()

mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
errors = (y_test - y_test_pred).flatten()

pct_01 = (np.abs(errors) < 0.01).mean() * 100
pct_05 = (np.abs(errors) < 0.05).mean() * 100

print(f"Test Set Evaluation (N = {len(y_test)}):")
print(f"MSE:  {mse:.6e} mJ^2")
print(f"RMSE: {rmse:.4f} mJ")
print(f"MAE:  {mae:.4f} mJ")
print(f"R^2:  {r2:.5f}")
print(f"|e| < 0.01 mJ: {pct_01:.2f}%")
print(f"|e| < 0.05 mJ: {pct_05:.2f}%")

# 7. Generate Figure (a) - Training & Validation Loss
plt.figure(figsize=(6.5, 4.8))
epochs_arr = np.arange(1, 31)
plt.plot(epochs_arr, train_losses, label='Training Loss', color='#1F618D', linewidth=2.2)
plt.plot(epochs_arr, val_losses, label='Validation Loss', color='#E74C3C', linewidth=2.0, linestyle='--')
plt.yscale('log')
plt.title('MLP Training and Validation Loss Convergence', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Epochs', fontsize=11, fontweight='bold')
plt.ylabel(r'MSE Loss ($\mathrm{mJ}^2$, log scale)', fontsize=11, fontweight='bold')
plt.legend(fontsize=11, framealpha=0.95)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/home/cristofe/Documents/Project/MLP/figuras/mlp_loss.png', dpi=300)
plt.close()

# 8. Generate Figure (b) - Test Residuals
plt.figure(figsize=(6.5, 4.8))
plt.hist(errors, bins=35, color='#8E44AD', edgecolor='black', alpha=0.75, label=f'Test Residuals ($N={len(errors)}$)')
plt.axvline(0, color='red', linestyle='--', linewidth=2.0, label=r'Zero Error ($y = \hat{y}$)')
plt.title(r'Test-Set Residual Error Distribution ($y - \hat{y}$)', fontsize=12, fontweight='bold', pad=10)
plt.xlabel(r'$y - \hat{y}$ (Prediction Error in mJ)', fontsize=11, fontweight='bold')
plt.ylabel('Sample Count (Test Set)', fontsize=11, fontweight='bold')
plt.legend(fontsize=11, framealpha=0.95)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/home/cristofe/Documents/Project/MLP/figuras/residuos.png', dpi=300)
plt.close()

print("Figures successfully generated in figuras/mlp_loss.png and figuras/residuos.png!")
