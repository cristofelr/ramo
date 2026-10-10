import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import scipy.ndimage as ndimage
import glob

print("Carregando logs de coordenadas físicas...")
node_file = glob.glob('/home/cristofe/Documents/Project/MLP/recent_logs/*-node.txt')[0]
nodes = pd.read_csv(node_file, sep=' ', header=None, names=['x', 'y', 'node_id'])

x = nodes['x'].values
y = nodes['y'].values

plt.figure(figsize=(12, 10))

print("Calculando o KDE espacial...")
sns.kdeplot(
    data=nodes,
    x='x',
    y='y',
    fill=True,
    cmap='viridis',
    levels=100,
    thresh=0,
    cbar=True,
    cbar_kws={'label': 'Densidade Física Estimada'}
)

plt.scatter(x, y, color='white', alpha=0.5, s=8, label='Sensores')
plt.scatter([0], [0], color='red', s=250, marker='^', zorder=10, label='Gateway')

# Encontrar os picos para ver quantos hotspots existem nessa simulação
xmin, xmax = -2000, 2000
ymin, ymax = -2000, 2000

X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([x, y])

kernel = gaussian_kde(values, bw_method=0.15) 
Z = np.reshape(kernel(positions).T, X.shape)

neighborhood_size = 20
threshold = 0.05 * Z.max()

data_max = ndimage.maximum_filter(Z, neighborhood_size)
maxima = (Z == data_max) & (Z > threshold)
peak_indices = np.where(maxima)

num_clusters = len(peak_indices[0])
print(f"Foram encontrados {num_clusters} hotspots físicos (clusters) nesta simulação!")

for i in range(num_clusters):
    px = X[peak_indices[0][i], peak_indices[1][i]]
    py = Y[peak_indices[0][i], peak_indices[1][i]]
    plt.scatter(px, py, color='yellow', s=200, edgecolor='black', zorder=15, marker='*')
    plt.annotate(f'H{i+1}', 
                 (px, py), 
                 textcoords="offset points", 
                 xytext=(0,12), 
                 ha='center', 
                 color='yellow', 
                 bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.7),
                 weight='bold')

plt.title(f'KDE Espacial (Coordenadas Reais) - {num_clusters} Hotspots Detectados')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.xlim(-2000, 2000)
plt.ylim(-2000, 2000)
plt.legend()
plt.grid(alpha=0.2)

out_img = '/home/cristofe/Documents/Project/MLP/kde_fisico.png'
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Mapa físico salvo em {out_img}")
