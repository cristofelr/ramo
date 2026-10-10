import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import glob

print("Lendo o arquivo node.txt da pasta recent_logs...")
node_file = glob.glob('/home/cristofe/Documents/Project/MLP/recent_logs/*node.txt')[0]
nodes = pd.read_csv(node_file, sep=' ', header=None, names=['x', 'y', 'node_id'])

x = nodes['x'].values
y = nodes['y'].values
values = np.vstack([x, y])

print("Treinando o modelo KDE Espacial com as coordenadas reais...")
# Usamos a mesma banda (0.15) que achou perfeitamente os 3 hotspots
kernel = gaussian_kde(values, bw_method=0.15) 

print("Atribuindo o valor exato de densidade para cada nó...")
# kernel(values) calcula a densidade de probabilidade para cada par (x, y) que foi passado
densities = kernel(values)

nodes['density'] = densities

out_file = '/home/cristofe/Documents/Project/MLP/recent_logs/node_with_density.csv'
nodes.to_csv(out_file, index=False)

print(f"\n====================================")
print(f"Arquivo gerado com sucesso em: {out_file}")
print(f"====================================")
print("Prévia das 5 primeiras linhas do novo dataset:")
print(nodes.head())
