import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Verifica se a GPU (RTX 5060) está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando o dispositivo: {device}")
if device.type == "cuda":
    print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")

# Define a arquitetura da Multilayer Perceptron (MLP)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        # Primeira camada oculta
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        # Segunda camada oculta
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        # Camada de saída
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# --- PARÂMETROS DA REDE ---
INPUT_SIZE = 8  # 8 features ambientais e de rede selecionadas do seu dataset
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
OUTPUT_SIZE = 1  # 1 para regressão/classificação binária, ou número de classes para multiclasse
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
EPOCHS = 100

# Inicializa o modelo e aloca na GPU (RTX 5060)
model = MLP(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE).to(device)

# Função de perda e otimizador
# - Regressão (ex: prever consumo de energia em mJ): nn.MSELoss()
# - Classificação binária: nn.BCEWithLogitsLoss()
# - Classificação multiclasse: nn.CrossEntropyLoss()
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- CARREGANDO DADOS ---
print("Carregando dataset_improved.csv...")
df = pd.read_csv("/home/cristofe/Documents/Project/MLP/dataset_improved.csv")

# Limpa linhas vazias/corrompidas (NaN) que fazem os gráficos e a rede explodirem
df.dropna(subset=['temperature', 'humidity', 'payload_size_bytes', 'rssi', 'snr', 'spreading_factor', 'bandwidth', 'battery_percentage', 'energy_mj'], inplace=True)

# Selecionar as features (X). Escolhi as métricas mais relevantes do seu CSV.
features = ['temperature', 'humidity', 'payload_size_bytes', 'rssi', 'snr', 'spreading_factor', 'bandwidth', 'battery_percentage']
X_data_raw = df[features].values.astype(np.float32)

# O alvo que a MLP precisa aprender a prever é o gasto de energia
y_data = df['energy_mj'].values.reshape(-1, 1).astype(np.float32)

# Normalização (Z-score) das variáveis. Sem isso, números gigantes como 'bandwidth' esmagam números pequenos como 'snr'
X_mean = np.mean(X_data_raw, axis=0)
X_std = np.std(X_data_raw, axis=0)
X_data = (X_data_raw - X_mean) / (X_std + 1e-8)

print(f"Dataset carregado com sucesso! {len(X_data)} registros prontos.")

# Converte para Tensores do PyTorch
X_tensor = torch.tensor(X_data)
y_tensor = torch.tensor(y_data)

# Cria o DataLoader para iterar em lotes (batches)
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- LOOP DE TREINAMENTO ---
print("\nIniciando o treinamento...")
history_loss = [] # Lista para guardar os erros de cada época

start_time_total = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train() # Coloca o modelo em modo de treino
    epoch_loss = 0.0
    
    for batch_X, batch_y in dataloader:
        # 1. Envia os lotes para a GPU
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 2. Forward pass (previsões)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 3. Backward pass e otimização
        optimizer.zero_grad() # Limpa os gradientes anteriores
        loss.backward()       # Calcula os gradientes na GPU
        optimizer.step()      # Atualiza os pesos
        
        epoch_loss += loss.item()
        
    # Calcula a média de erro na época e salva na lista
    avg_loss = epoch_loss / len(dataloader)
    history_loss.append(avg_loss)
    
    epoch_duration = time.time() - epoch_start_time
    
    # Imprime o progresso a cada 10 épocas
    if (epoch+1) % 10 == 0:
        # Pega a taxa de aprendizagem atual do otimizador
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Época [{epoch+1:3d}/{EPOCHS}] | Loss: {avg_loss:.5f} | LR: {current_lr:.5f} | Tempo/Época: {epoch_duration:.2f}s')

total_time = time.time() - start_time_total
print(f"\nTreinamento concluído em {total_time:.2f} segundos!")

# --- GERANDO O GRÁFICO DE PERDA (LOSS) ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), history_loss, color='b', label='Erro de Treinamento (MSE)')
plt.title('Curva de Aprendizado da MLP')
plt.xlabel('Épocas')
plt.ylabel('Erro (Loss)')
plt.grid(True)
plt.legend()
# Salva o gráfico como imagem na mesma pasta
plt.savefig('/home/cristofe/Documents/Project/MLP/grafico_loss.png')
print("Gráfico de convergência salvo como 'grafico_loss.png'!")

# --- AVALIAÇÃO GERAL E NOVOS GRÁFICOS ---
model.eval() # Coloca o modelo em modo de avaliação

with torch.no_grad():
    # Passa todos os dados normalizados para a GPU de uma vez
    X_tensor_all = torch.tensor(X_data).to(device)
    previsoes_all = model(X_tensor_all).cpu().numpy()
    
    # "Achata" as matrizes para listas simples (arrays 1D) para os gráficos
    todas_previsoes = previsoes_all.flatten()
    todos_reais = y_data.flatten()

# 1. Gráfico de Dispersão (Real vs Previsto)
plt.figure(figsize=(8, 8))
plt.scatter(todos_reais, todas_previsoes, alpha=0.3, color='green')

# Adicionando uma linha diagonal (representa a previsão perfeita)
max_val = max(np.max(todos_reais), np.max(todas_previsoes))
min_val = min(np.min(todos_reais), np.min(todas_previsoes))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Previsão Perfeita')

plt.title('Valores Reais vs. Previsões da MLP (Consumo de Energia)')
plt.xlabel('Energia Real (mJ)')
plt.ylabel('Energia Prevista (mJ)')
plt.legend()
plt.grid(True)
plt.savefig('/home/cristofe/Documents/Project/MLP/grafico_dispersao.png')
plt.close() # Limpa a memória do gráfico
print("Gráfico de Dispersão salvo como 'grafico_dispersao.png'!")

# 2. Histograma de Resíduos (Erros)
# Calcula a diferença entre o que a rede chutou e a realidade
residuos = todas_previsoes - todos_reais

plt.figure(figsize=(10, 6))
plt.hist(residuos, bins=50, color='purple', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='r', linestyle='dashed', linewidth=2, label='Erro Zero (Perfeito)')

plt.title('Distribuição dos Erros (Resíduos)')
plt.xlabel('Erro (Previsto - Real) em mJ')
plt.ylabel('Frequência (Quantidade de Pacotes)')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.savefig('/home/cristofe/Documents/Project/MLP/grafico_residuos.png')
plt.close() # Limpa a memória do gráfico
print("Histograma de Resíduos salvo como 'grafico_residuos.png'!")

# Mantendo o teste prático isolado no console
print(f"\n--- TESTE PRÁTICO (Primeira linha do CSV) ---")
print(f"Energia Prevista pela MLP: {todas_previsoes[0]:.4f} mJ")
print(f"Energia Real do Sensor:    {todos_reais[0]:.4f} mJ")
