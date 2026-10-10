import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print("Carregando datasets e recriando K-Means para extrair os Weak Centroids...")
df_nodes = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/node_with_density.csv')
df_links = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/links.csv')

# O K_opt e a semente inicial do KDE
k_opt = 3
seeds = df_nodes.nlargest(k_opt, 'density')[['x', 'y']].values
kmeans = KMeans(n_clusters=k_opt, init=seeds, n_init=1, random_state=42)
X = df_nodes[['x', 'y']].values
df_nodes['cluster_id'] = kmeans.fit_predict(X)

# --- AQUI ESTÁ O WEAK CENTROID ---
# O Weak Centroid (WC) é o "centro ideal" do cluster gerado pelo K-Means.
# Quase nunca existe um nó perfeitamente localizado em cima dele.
weak_centroids = kmeans.cluster_centers_

# Calculando a Energia Residual exata
energy_consumed = df_links.groupby('node_id')['energy_j'].sum().reset_index()
df_nodes = df_nodes.merge(energy_consumed, on='node_id', how='left')
df_nodes['energy_j'] = df_nodes['energy_j'].fillna(0) 
df_nodes['residual_energy'] = 1.0 - df_nodes['energy_j']
df_nodes['residual_energy'] = df_nodes['residual_energy'].clip(lower=0)

print("\n--- APLICANDO RAMO COM CÁLCULO EUCLIDIANO DO WEAK CENTROID ---")
# 1. Calcular a distância euclidiana real de cada nó para o seu Weak Centroid (WC)
distances = []
for idx, row in df_nodes.iterrows():
    wc_x, wc_y = weak_centroids[int(row['cluster_id'])]
    dist = np.sqrt((row['x'] - wc_x)**2 + (row['y'] - wc_y)**2)
    distances.append(dist)
    
df_nodes['dist_to_wc'] = distances

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# 2. O Fitness Score do RAMO:
# - Normaliza a energia
# - Normaliza a distância. Como queremos a MENOR distância, nós invertemos o score
# de proximidade (1.0 - dist_norm), para que a proximidade some pontos positivos!
df_nodes['norm_proximity'] = 1.0 - min_max_normalize(df_nodes['dist_to_wc'])
df_nodes['norm_energy'] = min_max_normalize(df_nodes['residual_energy'])

# Pesos equilibrados: 50% de peso em quão próximo está do WC, 50% em Energia!
w_proximity = 0.5
w_energy = 0.5

df_nodes['ramo_score'] = (w_proximity * df_nodes['norm_proximity']) + (w_energy * df_nodes['norm_energy'])

cluster_heads = []
for cid in range(k_opt):
    cluster_nodes = df_nodes[df_nodes['cluster_id'] == cid]
    # RAMO seleciona o que tem o score máximo (Maior Energia + Menor Distância ao WC)
    ch = cluster_nodes.loc[cluster_nodes['ramo_score'].idxmax()]
    cluster_heads.append(ch)
    print(f"Cluster {cid} -> CH Eleito: Node {int(ch['node_id'])}")
    print(f"   [ RAMO Score: {ch['ramo_score']:.4f} | Dist. ao WC: {ch['dist_to_wc']:.1f}m | E_Res: {ch['residual_energy']:.4f}J ]")

df_ch = pd.DataFrame(cluster_heads)

print("\nDesenhando o mapa das linhas de Trade-off (WC -> CH)...")
plt.figure(figsize=(12, 10))
colors = ['#00d4ff', '#ff007b', '#00ff66']

for i in range(k_opt):
    c_pts = df_nodes[df_nodes['cluster_id'] == i]
    plt.scatter(c_pts['x'], c_pts['y'], c=colors[i], label=f'Cluster {i} Nodes', alpha=0.2, s=20)

# Plotar as Cruzes Cinzas (Os Weak Centroids teóricos)
plt.scatter(weak_centroids[:, 0], weak_centroids[:, 1], c='gray', marker='P', s=400, edgecolor='white', zorder=5, label='Weak Centroids (Ideais)')

# Plotar as Estrelas (Os CHs Reais Eleitos)
plt.scatter(df_ch['x'], df_ch['y'], c='gold', marker='*', s=800, edgecolor='black', zorder=10, label='CHs Reais (RAMO)')

# Traçar a linha de Offset (O trade-off visual do RAMO)
for cid in range(k_opt):
    wc_x, wc_y = weak_centroids[cid]
    ch_x, ch_y = df_ch[df_ch['cluster_id'] == cid][['x', 'y']].values[0]
    plt.plot([wc_x, ch_x], [wc_y, ch_y], color='yellow', linestyle='--', linewidth=2, zorder=8)

# Anotações
for idx, row in df_ch.iterrows():
    plt.annotate(f"CH {int(row['node_id'])}", (row['x'], row['y']), textcoords="offset points", xytext=(0,20), ha='center', color='gold', bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.9), weight='bold', fontsize=14)

plt.scatter([0], [0], color='white', s=250, marker='^', zorder=15, label='Gateway')

plt.title('Seleção RAMO Multi-Objetivo (Weak Centroid vs Energia)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend(loc='lower left')
plt.grid(alpha=0.2)
plt.gca().set_facecolor('#1a1a2e')

out_img = '/home/cristofe/Documents/Project/MLP/recent_logs/ramo_weak_centroids.png'
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Mapa visual atualizado exportado em: {out_img}")
