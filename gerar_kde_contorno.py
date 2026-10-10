import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("Carregando o dataset...")
csv_path = '/home/cristofe/Documents/Project/MLP/dataset_improved_density.csv'
df = pd.read_csv(csv_path)

plt.figure(figsize=(12, 9))

print("Gerando o mapa de calor em formato de contorno para destacar os picos...")
# KDE em formato de relevo (Contorno 2D)
ax = sns.kdeplot(
    data=df,
    x='rssi',
    y='snr',
    fill=True,
    cmap='turbo', # Turbo ou inferno destacam muito bem os picos
    levels=100,
    thresh=0,
    cbar=True,
    cbar_kws={'label': 'Densidade (KDE)'}
)

# Coloca os pontos sutilmente por cima
plt.scatter(df['rssi'], df['snr'], color='black', alpha=0.05, s=2)

# Adiciona um marcador no pico exato da densidade
peak_idx = df['density'].idxmax()
peak_rssi = df.loc[peak_idx, 'rssi']
peak_snr = df.loc[peak_idx, 'snr']
peak_density = df.loc[peak_idx, 'density']

plt.scatter([peak_rssi], [peak_snr], color='white', s=150, edgecolor='black', zorder=5, marker='*')
plt.annotate(f'Pico Máximo\nRSSI: {peak_rssi:.1f}\nSNR: {peak_snr:.1f}', 
             (peak_rssi, peak_snr), 
             textcoords="offset points", 
             xytext=(0,15), 
             ha='center', 
             color='white', 
             bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="black", lw=1, alpha=0.7),
             weight='bold')

plt.title('Mapa de Calor de Densidade KDE (Contorno Real)')
plt.xlabel('RSSI (dBm)')
plt.ylabel('SNR (dB)')
plt.grid(alpha=0.3)

out_img = '/home/cristofe/Documents/Project/MLP/kde_heatmap.png'
plt.savefig(out_img, dpi=300, bbox_inches='tight')
print(f"Mapa de calor salvo em {out_img}")
