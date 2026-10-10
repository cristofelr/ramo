import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Verifica GPU (Acelerando na RTX)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Motor PyTorch iniciado! Usando hardware: {device}")

# Define a Topologia da MLP
class EnergyPredictorMLP(nn.Module):
    def __init__(self):
        super(EnergyPredictorMLP, self).__init__()
        # 5 Entradas (Spreading Factor, Bandwidth, RSSI, SNR, Time On Air)
        self.fc1 = nn.Linear(5, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        # 1 Saída (Previsão da Energia em Joules)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        return self.fc3(out)

# 1. CARREGAR OS DADOS REAIS DO LO-RA
print("\n[1/4] Carregando os logs de transmissão dos sensores...")
df = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/links.csv')
df.dropna(subset=['sf', 'bw', 'rssi', 'snr', 'toa', 'energy_j'], inplace=True)

# Separa Features e Target
features = ['sf', 'bw', 'rssi', 'snr', 'toa']
X_raw = df[features].values.astype(np.float32)
y_raw = df['energy_j'].values.reshape(-1, 1).astype(np.float32)

# Normalização de Variáveis
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X_data = (X_raw - X_mean) / (X_std + 1e-8)

dataset = TensorDataset(torch.tensor(X_data), torch.tensor(y_raw))
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

model = EnergyPredictorMLP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 2. TREINAR A REDE
print(f"\n[2/4] Treinando a MLP para aprender as leis físicas de consumo LoRaWAN...")
epochs = 30
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"      Época [{epoch+1}/{epochs}] | Erro (Loss): {loss.item():.6f}")

# 3. GERAR AS PREVISÕES (INFERÊNCIA)
print("\n[3/4] Aplicando a inferência em todos os pacotes da rede...")
model.eval()
with torch.no_grad():
    X_all = torch.tensor(X_data).to(device)
    preds = model(X_all).cpu().numpy().flatten()

# Adiciona a coluna com a previsão que a máquina "chutou" para cada pacote
df['predicted_energy_j'] = preds
# Corrige anomalias negativas da rede neural
df['predicted_energy_j'] = df['predicted_energy_j'].clip(lower=0)

# 4. CONSOLIDAR O GASTO POR SENSOR E SALVAR
print("\n[4/4] Agrupando o gasto de energia por Sensor ID e exportando resultados...")
# Agrupamos por sensor e somamos todas as energias previstas de todos os pacotes dele
df_pred = df.groupby('node_id')['predicted_energy_j'].sum().reset_index()

out_file = '/home/cristofe/Documents/Project/MLP/recent_logs/mlp_predicted_energy.csv'
df_pred.to_csv(out_file, index=False)

print(f"\n=================================================")
print(f"Sucesso! Previsões de bateria exportadas para:")
print(f"-> {out_file}")
print("=================================================")
print("Tabela Gerada (Primeiras 5 linhas):")
print(df_pred.head())
