import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse
import sys
import os

# Configurar argumentos via linha de comando
parser = argparse.ArgumentParser(description='RAMO Algoritmo: Seleção de CHs por 4 Objetivos e IA')
parser.add_argument('-d', '--dist', type=float, default=0.25, help='Peso da Distância ao Weak Centroid')
parser.add_argument('-e', '--energy', type=float, default=0.25, help='Peso da Energia')
parser.add_argument('-k', '--kde', type=float, default=0.25, help='Peso da Densidade Espacial KDE')
parser.add_argument('-g', '--gw', type=float, default=0.25, help='Peso da Distância ao Gateway')
parser.add_argument('-f', '--file', type=str, default='/home/cristofe/Documents/Project/MLP/recent_logs/node_with_clusters.csv', help='Arquivo CSV contendo os nós da rede')
parser.add_argument('-b', '--battery', type=str, choices=['residual', 'predita'], default='residual', help='Fonte do gasto de energia: "residual" (log NS-3) ou "predita" (arquivo da MLP)')
args = parser.parse_args()

w1, w2, w3, w4 = args.dist, args.energy, args.kde, args.gw

total = w1 + w2 + w3 + w4
if total == 0:
    print("Os pesos não podem ser nulos.")
    sys.exit(1)
w1, w2, w3, w4 = w1/total, w2/total, w3/total, w4/total

print("=========================================")
print(f"Executando RAMO Quadri-Objetivo")
print(f"Fonte da Bateria: {args.battery.upper()}")
print(f"Pesos -> Weak Centroid: {w1*100:.1f}% | Energia: {w2*100:.1f}% | Densidade: {w3*100:.1f}% | Gateway: {w4*100:.1f}%")
print("=========================================\n")

if not os.path.exists(args.file):
    print(f"Erro: O arquivo {args.file} não foi encontrado!")
    sys.exit(1)

df_nodes = pd.read_csv(args.file)

if 'cluster_id' in df_nodes.columns:
    print("Clusters pré-existentes detectados! Calculando os Weak Centroids a partir deles...")
    weak_centroids = []
    k_opt = df_nodes['cluster_id'].nunique()
    for cid in range(k_opt):
        c_nodes = df_nodes[df_nodes['cluster_id'] == cid]
        wc_x = c_nodes['x'].mean()
        wc_y = c_nodes['y'].mean()
        weak_centroids.append([wc_x, wc_y])
    weak_centroids = np.array(weak_centroids)
else:
    print("Dataset não está clusterizado. Aplicando K-Means Interno...")
    k_opt = 3
    seeds = df_nodes.nlargest(k_opt, 'density')[['x', 'y']].values
    kmeans = KMeans(n_clusters=k_opt, init=seeds, n_init=1, random_state=42)
    X = df_nodes[['x', 'y']].values
    df_nodes['cluster_id'] = kmeans.fit_predict(X)
    weak_centroids = kmeans.cluster_centers_

# --- LÓGICA DO NOVO PARÂMETRO -b (BATTERY) ---
if args.battery == 'predita':
    print("Carregando o desgaste energético PREVISTO pela rede neural MLP...")
    pred_file = '/home/cristofe/Documents/Project/MLP/recent_logs/mlp_predicted_energy.csv'
    if not os.path.exists(pred_file):
        print(f"Erro: O arquivo de predição {pred_file} não existe! Rode a MLP antes.")
        sys.exit(1)
    df_energy = pd.read_csv(pred_file)
    # Padroniza a coluna para facilitar o Merge
    df_energy.rename(columns={'predicted_energy_j': 'energy_consumed'}, inplace=True)
else:
    print("Calculando o desgaste energético REAL (histórico do NS-3)...")
    df_links = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/links.csv')
    df_energy = df_links.groupby('node_id')['energy_j'].sum().reset_index()
    df_energy.rename(columns={'energy_j': 'energy_consumed'}, inplace=True)

# Merge com as coordenadas
df_nodes = df_nodes.merge(df_energy, on='node_id', how='left')
df_nodes['energy_consumed'] = df_nodes['energy_consumed'].fillna(0) 

# Calcula a energia restante (Inicia com 1.0 Joule)
df_nodes['residual_energy'] = 1.0 - df_nodes['energy_consumed']
df_nodes['residual_energy'] = df_nodes['residual_energy'].clip(lower=0)

distances_wc = []
distances_gw = []
gw_x, gw_y = 0, 0 

for idx, row in df_nodes.iterrows():
    wc_x, wc_y = weak_centroids[int(row['cluster_id'])]
    dist_wc = np.sqrt((row['x'] - wc_x)**2 + (row['y'] - wc_y)**2)
    dist_gw = np.sqrt((row['x'] - gw_x)**2 + (row['y'] - gw_y)**2)
    distances_wc.append(dist_wc)
    distances_gw.append(dist_gw)
    
df_nodes['dist_to_wc'] = distances_wc
df_nodes['dist_to_gw'] = distances_gw

def min_max_normalize(series):
    if series.max() == series.min():
        return pd.Series(1, index=series.index)
    return (series - series.min()) / (series.max() - series.min())

df_nodes['norm_proximity'] = 1.0 - min_max_normalize(df_nodes['dist_to_wc'])
df_nodes['norm_gw_proximity'] = 1.0 - min_max_normalize(df_nodes['dist_to_gw'])
df_nodes['norm_energy'] = min_max_normalize(df_nodes['residual_energy'])
df_nodes['norm_density'] = min_max_normalize(df_nodes['density'])

df_nodes['ramo_score'] = (w1 * df_nodes['norm_proximity']) + (w2 * df_nodes['norm_energy']) + (w3 * df_nodes['norm_density']) + (w4 * df_nodes['norm_gw_proximity'])

cluster_heads = []
for cid in range(k_opt):
    cluster_nodes = df_nodes[df_nodes['cluster_id'] == cid]
    ch = cluster_nodes.loc[cluster_nodes['ramo_score'].idxmax()]
    cluster_heads.append(ch)
    print(f"Cluster {cid} -> CH Eleito: Node {int(ch['node_id'])}")
    print(f"   [ Score: {ch['ramo_score']:.4f} | E_Res ({args.battery.upper()}): {ch['residual_energy']:.4f}J | Dist. WC: {ch['dist_to_wc']:.1f}m ]")

df_ch = pd.DataFrame(cluster_heads)

plt.figure(figsize=(12, 10))
colors = ['#00d4ff', '#ff007b', '#00ff66', '#ff0000', '#ffff00']

for i in range(k_opt):
    c_pts = df_nodes[df_nodes['cluster_id'] == i]
    plt.scatter(c_pts['x'], c_pts['y'], c=colors[i % len(colors)], label=f'Cluster {i} Nodes', alpha=0.3, s=20)

plt.scatter(weak_centroids[:, 0], weak_centroids[:, 1], c='gray', marker='P', s=400, edgecolor='white', zorder=5, label='Weak Centroids (Geométricos)')
plt.scatter(df_ch['x'], df_ch['y'], c='gold', marker='*', s=800, edgecolor='black', zorder=10, label=f'CHs (Bat: {args.battery})')

for cid in range(k_opt):
    wc_x, wc_y = weak_centroids[cid]
    ch_x, ch_y = df_ch[df_ch['cluster_id'] == cid][['x', 'y']].values[0]
    plt.plot([wc_x, ch_x], [wc_y, ch_y], color='yellow', linestyle='--', linewidth=2, zorder=8)
    plt.plot([0, ch_x], [0, ch_y], color='cyan', linestyle=':', linewidth=1.5, zorder=7, alpha=0.7)

for idx, row in df_ch.iterrows():
    plt.annotate(f"CH {int(row['node_id'])}", (row['x'], row['y']), textcoords="offset points", xytext=(0,20), ha='center', color='gold', bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.9), weight='bold', fontsize=14)

plt.scatter([0], [0], color='white', s=250, marker='^', zorder=15, label='Gateway (0,0)')

plt.title(f'RAMO (Bat: {args.battery.capitalize()}): WC={w1*100:.0f}% | Ener={w2*100:.0f}% | Dens={w3*100:.0f}% | GW={w4*100:.0f}%')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend(loc='lower left')
plt.grid(alpha=0.2)
plt.gca().set_facecolor('#1a1a2e')

out_img = '/home/cristofe/Documents/Project/MLP/recent_logs/ramo_quadri_objective.png'
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"\nMapa visual exportado em: {out_img}")
