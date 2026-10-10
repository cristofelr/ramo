import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import scipy.ndimage as ndimage
import glob
import os

print("Buscando o arquivo de coordenadas na pasta recent_logs...")
node_file = glob.glob('/home/cristofe/Documents/Project/MLP/recent_logs/*node.txt')[0]
nodes = pd.read_csv(node_file, sep=' ', header=None, names=['x', 'y', 'node_id'])

x = nodes['x'].values
y = nodes['y'].values

# O ERRO ANTERIOR FOI AQUI: A rede tem R=5000m. 
# O código antigo limitava a visão do mapa entre -2000 e 2000, 
# "cortando" fora os outros 3 hotspots! Vamos fazer o limite dinâmico.
margin = 500
xmin, xmax = x.min() - margin, x.max() + margin
ymin, ymax = y.min() - margin, y.max() + margin

print(f"Dimensões do grid expandidas para suportar a simulação R=5000:")
print(f"X vai de {xmin:.0f} a {xmax:.0f} | Y vai de {ymin:.0f} a {ymax:.0f}")

X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([x, y])

print("Aplicando o KDE para calcular a densidade...")
kernel = gaussian_kde(values, bw_method=0.15) 
Z = np.reshape(kernel(positions).T, X.shape)

print("Realizando a caça aos Picos (K-Estimation)...")
neighborhood_size = 20
threshold = 0.05 * Z.max()

data_max = ndimage.maximum_filter(Z, neighborhood_size)
maxima = (Z == data_max) & (Z > threshold)
peak_indices = np.where(maxima)

k_opt = len(peak_indices[0])
print(f"==========================================")
print(f"K_OPT DETECTADO: {k_opt} HOTSPOTS ENCONTRADOS")
print(f"==========================================")

plt.figure(figsize=(12, 10))
plt.contourf(X, Y, Z, levels=100, cmap='turbo')
plt.colorbar(label='Densidade (KDE)')

plt.scatter(x, y, color='black', alpha=0.3, s=8, label='Sensores')
plt.scatter([0], [0], color='red', s=250, marker='^', zorder=10, label='Gateway')

# Plota os centros K_opt
for i in range(k_opt):
    px = X[peak_indices[0][i], peak_indices[1][i]]
    py = Y[peak_indices[0][i], peak_indices[1][i]]
    plt.scatter(px, py, color='lime', s=350, edgecolor='black', zorder=15, marker='*')
    plt.annotate(f'K{i+1}', 
                 (px, py), 
                 textcoords="offset points", 
                 xytext=(0,15), 
                 ha='center', 
                 color='lime', 
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.8),
                 weight='bold', fontsize=14)

plt.title(f'E-DBC K-Estimation: K_opt = {k_opt} clusters físicos ideais')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend(loc='upper right')
plt.grid(alpha=0.2)

out_img = '/home/cristofe/Documents/Project/MLP/recent_logs/k_opt_fisico.png'
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Mapa salvo em {out_img}")
