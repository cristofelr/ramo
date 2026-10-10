import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde

print("Carregando datasets...")
df_nodes = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/node_with_density.csv')
df_links = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/links.csv')

print("Calculando energia residual (para ser usada como gravidade)...")
energy_consumed = df_links.groupby('node_id')['energy_j'].sum().reset_index()
df_nodes = df_nodes.merge(energy_consumed, on='node_id', how='left')
df_nodes['energy_j'] = df_nodes['energy_j'].fillna(0) 
df_nodes['residual_energy'] = 1.0 - df_nodes['energy_j']
df_nodes['residual_energy'] = df_nodes['residual_energy'].clip(lower=0)

# A MÁGICA: Definimos o "Peso Gravitacional" de cada nó de forma INVERSAMENTE
# proporcional a sua bateria restante. Ou seja, quem tem MENOS bateria "pesa" MAIS.
df_nodes['gravity_weight'] = 1.0 / (df_nodes['residual_energy'] + 0.05)

print("\n--- K-MEANS PONDERADO PELA BATERIA DOS NÓS ---")
x = df_nodes['x'].values
y = df_nodes['y'].values
X = df_nodes[['x', 'y']].values

k_opt = 3

print("Calculando o KDE ponderado pela bateria para gerar sementes puxadas pros fracos...")
values = np.vstack([x, y])
kernel = gaussian_kde(values, bw_method=0.15, weights=df_nodes['gravity_weight'].values)
df_nodes['density_weighted'] = kernel(values)

seeds = df_nodes.nlargest(k_opt, 'density_weighted')[['x', 'y']].values

print("Arrastando os Weak Centroids do K-Means para perto dos mais fracos...")
kmeans = KMeans(n_clusters=k_opt, init=seeds, n_init=1, random_state=42)
# sample_weight arrasta o centro geométrico do cluster para onde tem baterias piores!
df_nodes['cluster_id'] = kmeans.fit_predict(X, sample_weight=df_nodes['gravity_weight'].values)

weak_centroids = kmeans.cluster_centers_

print("\n--- APLICANDO RAMO ---")
# Agora que o WC mudou de lugar, o RAMO continua buscando um CH que seja forte
# mas próximo desse novo WC que está no "leito de morte" dos nós vizinhos.
distances = []
for idx, row in df_nodes.iterrows():
    wc_x, wc_y = weak_centroids[int(row['cluster_id'])]
    dist = np.sqrt((row['x'] - wc_x)**2 + (row['y'] - wc_y)**2)
    distances.append(dist)
    
df_nodes['dist_to_wc'] = distances

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df_nodes['norm_proximity'] = 1.0 - min_max_normalize(df_nodes['dist_to_wc'])
df_nodes['norm_energy'] = min_max_normalize(df_nodes['residual_energy'])

# Dando muito mais peso (85%) para a proximidade. Assim o CH é fortemente
# puxado para ficar o mais colado possível no Weak Centroid!
df_nodes['ramo_score'] = (0.85 * df_nodes['norm_proximity']) + (0.15 * df_nodes['norm_energy'])

cluster_heads = []
for cid in range(k_opt):
    cluster_nodes = df_nodes[df_nodes['cluster_id'] == cid]
    ch = cluster_nodes.loc[cluster_nodes['ramo_score'].idxmax()]
    cluster_heads.append(ch)
    print(f"Cluster {cid} -> Novo CH Eleito: Node {int(ch['node_id'])}")

df_ch = pd.DataFrame(cluster_heads)

print("\nDesenhando mapa visual focado no gasto de energia...")
plt.figure(figsize=(12, 10))
colors = ['#00d4ff', '#ff007b', '#00ff66']

# Os sensores no mapa ficarão visualmente MAIORES quanto MAIS energia tiverem gastado
# (Representação visual dos pontos de socorro que arrastam o WC)
node_sizes = (1.0 - df_nodes['residual_energy']) * 300 + 10

for i in range(k_opt):
    c_pts = df_nodes[df_nodes['cluster_id'] == i]
    sizes = node_sizes[c_pts.index]
    plt.scatter(c_pts['x'], c_pts['y'], c=colors[i], label=f'Cluster {i} Nodes', alpha=0.4, s=sizes)

plt.scatter(weak_centroids[:, 0], weak_centroids[:, 1], c='gray', marker='P', s=400, edgecolor='white', zorder=5, label='Weighted Weak Centroids')
plt.scatter(df_ch['x'], df_ch['y'], c='gold', marker='*', s=800, edgecolor='black', zorder=10, label='RAMO CHs')

for cid in range(k_opt):
    wc_x, wc_y = weak_centroids[cid]
    ch_x, ch_y = df_ch[df_ch['cluster_id'] == cid][['x', 'y']].values[0]
    plt.plot([wc_x, ch_x], [wc_y, ch_y], color='yellow', linestyle='--', linewidth=2, zorder=8)

for idx, row in df_ch.iterrows():
    plt.annotate(f"CH {int(row['node_id'])}", (row['x'], row['y']), textcoords="offset points", xytext=(0,20), ha='center', color='gold', bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.9), weight='bold', fontsize=14)

plt.scatter([0], [0], color='white', s=250, marker='^', zorder=15, label='Gateway')

plt.title('Energy-Weighted K-Means e RAMO CH Selection')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend(loc='lower left')
plt.grid(alpha=0.2)
plt.gca().set_facecolor('#1a1a2e')

out_img = '/home/cristofe/Documents/Project/MLP/recent_logs/ramo_battery_weighted.png'
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Mapa visual atualizado exportado em: {out_img}")
