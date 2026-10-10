import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

print("Carregando o dataset...")
csv_path = '/home/cristofe/Documents/Project/MLP/dataset_improved.csv'
df = pd.read_csv(csv_path)

# Usando RSSI e SNR como proxy para a localização/área no KDE, já que não temos coordenadas GPS
col_x = 'rssi'
col_y = 'snr'

print(f"Filtrando dados nulos em {col_x} e {col_y}...")
df_clean = df.dropna(subset=[col_x, col_y]).copy()

x = df_clean[col_x].values
y = df_clean[col_y].values

print("Calculando a densidade com KDE (pode levar alguns segundos)...")
# Agrupando x e y para o KDE
xy = np.vstack([x, y])
kde = gaussian_kde(xy)

# Avaliando a densidade em cada ponto
density = kde(xy)
df_clean['density'] = density

# Salvando os dados com a densidade
out_csv = '/home/cristofe/Documents/Project/MLP/dataset_improved_density.csv'
df_clean.to_csv(out_csv, index=False)
print(f"Dataset com densidade salvo em {out_csv}")

print("Gerando o mapa de calor...")
plt.figure(figsize=(10, 8))

# Ordena os pontos pela densidade para que os mais densos fiquem por cima
idx = density.argsort()
x, y, density = x[idx], y[idx], density[idx]

scatter = plt.scatter(x, y, c=density, s=20, cmap='inferno', edgecolor='none')
plt.colorbar(scatter, label='Densidade Estimada (KDE)')

plt.title('Mapa de Calor da Densidade (KDE) - RSSI vs SNR')
plt.xlabel('RSSI (dBm)')
plt.ylabel('SNR (dB)')
plt.grid(alpha=0.3)

out_img = '/home/cristofe/Documents/Project/MLP/kde_heatmap.png'
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Mapa de calor salvo em {out_img}")
