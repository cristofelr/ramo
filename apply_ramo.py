import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Carregando os dados de Nós, Clusters e Pacotes...")
df_nodes = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/node_with_clusters.csv')
df_links = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/links.csv')

print("Calculando a Energia Residual atual de cada sensor...")
# Soma toda a energia gasta por nó nas transmissões
energy_consumed = df_links.groupby('node_id')['energy_j'].sum().reset_index()

# Junta essa informação com as coordenadas/densidades
df_nodes = df_nodes.merge(energy_consumed, on='node_id', how='left')
# Se o nó não enviou nada no log, gasto = 0
df_nodes['energy_j'] = df_nodes['energy_j'].fillna(0) 

# Assumindo energia inicial de 1.0 Joule para todos (padrão NS-3/LoRaSimPlus)
initial_energy = 1.0 
df_nodes['residual_energy'] = initial_energy - df_nodes['energy_j']
df_nodes['residual_energy'] = df_nodes['residual_energy'].clip(lower=0)

print("\n--- APLICANDO O ALGORITMO MULTI-OBJETIVO (RAMO) ---")
# RAMO = Residual-energy and Area-density Multi-objective selection
# Normalizamos as grandezas para elas terem o mesmo peso na equação

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Pesos equilibrados: 50% densidade (garante cobertura), 50% energia (garante lifetime)
w_density = 0.5
w_energy = 0.5

df_nodes['norm_density'] = min_max_normalize(df_nodes['density'])
df_nodes['norm_energy'] = min_max_normalize(df_nodes['residual_energy'])

# Pontuação (Fitness) do RAMO
df_nodes['ramo_score'] = (w_density * df_nodes['norm_density']) + (w_energy * df_nodes['norm_energy'])

cluster_heads = []
for cid in sorted(df_nodes['cluster_id'].unique()):
    cluster_nodes = df_nodes[df_nodes['cluster_id'] == cid]
    # O Nó com maior RAMO Score dentro do cluster é eleito o CH!
    ch = cluster_nodes.loc[cluster_nodes['ramo_score'].idxmax()]
    cluster_heads.append(ch)
    print(f"Cluster {int(cid)} -> CH Eleito: Sensor ID {int(ch['node_id'])}")
    print(f"   [ RAMO Score: {ch['ramo_score']:.4f} | Energia Restante: {ch['residual_energy']:.4f}J | Densidade: {ch['density']:.6f} ]")

df_ch = pd.DataFrame(cluster_heads)

print("\nDesenhando a topologia final E-DBC com os Cluster Heads do RAMO...")
plt.figure(figsize=(12, 10))
colors = ['#00d4ff', '#ff007b', '#00ff66']

for i in range(3):
    c_pts = df_nodes[df_nodes['cluster_id'] == i]
    plt.scatter(c_pts['x'], c_pts['y'], c=colors[i], label=f'Cluster {i} (Nós)', alpha=0.3, s=20)
    
# Plotar os CHs Eleitos pelo RAMO
plt.scatter(df_ch['x'], df_ch['y'], c='gold', marker='*', s=800, edgecolor='white', zorder=10, label='RAMO Cluster Heads (CH)')

for idx, row in df_ch.iterrows():
    plt.annotate(f"CH {int(row['node_id'])}", 
                 (row['x'], row['y']), 
                 textcoords="offset points", 
                 xytext=(0,20), 
                 ha='center', 
                 color='gold', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.9),
                 weight='bold', fontsize=14)

plt.scatter([0], [0], color='white', s=250, marker='^', zorder=15, label='Gateway')

plt.title('RAMO Algoritmo: Seleção de Cluster Heads Otimizada (E-DBC)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend(loc='lower left')
plt.grid(alpha=0.2)
plt.gca().set_facecolor('#1a1a2e')

out_img = '/home/cristofe/Documents/Project/MLP/recent_logs/ramo_cluster_heads.png'
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Mapa final RAMO salvo em: {out_img}")
