import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy.ndimage as ndimage

print("Carregando o dataset...")
csv_path = '/home/cristofe/Documents/Project/MLP/dataset_improved_density.csv'
df = pd.read_csv(csv_path)

x = df['rssi'].values
y = df['snr'].values

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

print("Gerando Grid 200x200...")
X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([x, y])

print("Calculando o KDE...")
kernel = gaussian_kde(values, bw_method=0.10) 
Z = np.reshape(kernel(positions).T, X.shape)

print("Procurando picos locais...")
neighborhood_size = 10
threshold = 0.05 * Z.max()

data_max = ndimage.maximum_filter(Z, neighborhood_size)
maxima = (Z == data_max) & (Z > threshold)
peak_indices = np.where(maxima)

# Extrai os picos e suas densidades
peaks = []
for i in range(len(peak_indices[0])):
    idx_x = peak_indices[0][i]
    idx_y = peak_indices[1][i]
    density_val = Z[idx_x, idx_y]
    peaks.append((idx_x, idx_y, density_val))

# ORDENANDO: Pega apenas os 3 maiores picos em densidade!
peaks.sort(key=lambda p: p[2], reverse=True)
top_3_peaks = peaks[:3]

plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Z, levels=100, cmap='turbo')
plt.colorbar(label='Densidade Estimada (KDE)')

# Dados originais de fundo
plt.scatter(x, y, color='black', alpha=0.05, s=2)

print("Marcando os 3 principais picos (Centróides Iniciais/Weak Centroids) para o K-Means...")

for i, p in enumerate(top_3_peaks):
    px = X[p[0], p[1]]
    py = Y[p[0], p[1]]
    plt.scatter(px, py, color='white', s=250, edgecolor='black', zorder=5, marker='*')
    plt.annotate(f'K{i+1}', 
                 (px, py), 
                 textcoords="offset points", 
                 xytext=(0,15), 
                 ha='center', 
                 color='white', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.8),
                 weight='bold', fontsize=12)

plt.title('Estimativa K=3 (E-DBC): Top 3 Densidades Iniciais do RAMO')
plt.xlabel('RSSI (dBm)')
plt.ylabel('SNR (dB)')
plt.grid(alpha=0.3)

out_img = '/home/cristofe/Documents/Project/MLP/kde_3_peaks.png'
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Mapa salvo em {out_img}")
