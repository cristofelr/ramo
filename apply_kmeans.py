import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print("Carregando o dataset com coordenadas e densidade...")
df = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/node_with_density.csv')

# O K_opt calculado pelo E-DBC anteriormente foi 3
k_opt = 3

print(f"Buscando as {k_opt} sementes iniciais baseadas nos picos de densidade KDE...")
# No método E-DBC (RAMO), inicializamos o K-Means usando as áreas de maior densidade 
# como 'Weak Centroids' (sementes iniciais).
seeds = df.nlargest(k_opt, 'density')[['x', 'y']].values
print("Sementes selecionadas (X, Y):")
print(seeds)

print("Treinando o modelo K-Means...")
# Forçamos n_init=1 porque estamos passando sementes manuais baseadas no KDE
kmeans = KMeans(n_clusters=k_opt, init=seeds, n_init=1, random_state=42)

# O modelo k-means treina apenas em X e Y para definir os perímetros físicos
X = df[['x', 'y']].values
df['cluster_id'] = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

print("Salvando novo CSV com ID do Cluster...")
out_csv = '/home/cristofe/Documents/Project/MLP/recent_logs/node_with_clusters.csv'
df.to_csv(out_csv, index=False)

print("Gerando o gráfico final de clusterização...")
plt.figure(figsize=(10, 8))
colors = ['#00d4ff', '#ff007b', '#00ff66'] # Cores vibrantes (Ciano, Rosa Neon, Verde Neon)

# Plotamos cada cluster
for i in range(k_opt):
    cluster_points = df[df['cluster_id'] == i]
    plt.scatter(cluster_points['x'], cluster_points['y'], c=colors[i], label=f'Cluster {i}', alpha=0.6, s=20)
    
# Plotamos os Centroids oficiais finais definidos pelo K-Means
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=300, edgecolor='white', zorder=10, label='K-Means Final Centroids')

# Plotamos o Gateway
plt.scatter([0], [0], color='black', s=250, marker='^', zorder=15, label='Gateway')

plt.title('Clusterização K-Means Baseada em K-Estimation KDE (E-DBC)')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.grid(alpha=0.2)
# Define fundo levemente escuro para destacar as cores vibrantes dos clusters
plt.gca().set_facecolor('#1a1a2e')

out_img = '/home/cristofe/Documents/Project/MLP/recent_logs/kmeans_clusters.png'
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Gráfico K-Means exportado: {out_img}")
