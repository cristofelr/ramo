import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- PARÂMETROS FÍSICOS DA SIMULAÇÃO (LoRa Simplificado) ---
INITIAL_ENERGY = 1.0 # Joules de Bateria no momento zero
E_TX_BASE = 0.0005   # Custo base pra ligar o transmissor
E_AMP = 0.000005     # Custo de amplificação por metro percorrido pela onda

def calculate_tx_cost(distance):
    # Quanto mais longe, mais bateria gasta na Rodada
    return E_TX_BASE + (E_AMP * distance)

def run_simulation(mode="residual"):
    # Carrega a rede zerada
    df = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/node_with_clusters.csv')
    df['battery'] = INITIAL_ENERGY
    df['is_dead'] = False
    
    k_opt = df['cluster_id'].nunique()
    
    # 1. Pre-calcula Distancias e Weak Centroids Fixos
    weak_centroids = []
    for cid in range(k_opt):
        c_nodes = df[df['cluster_id'] == cid]
        weak_centroids.append([c_nodes['x'].mean(), c_nodes['y'].mean()])
    weak_centroids = np.array(weak_centroids)
    
    gw_x, gw_y = 0, 0
    df['dist_to_wc'] = [np.sqrt((row['x'] - weak_centroids[int(row['cluster_id'])][0])**2 + (row['y'] - weak_centroids[int(row['cluster_id'])][1])**2) for idx, row in df.iterrows()]
    df['dist_to_gw'] = [np.sqrt((row['x'] - gw_x)**2 + (row['y'] - gw_y)**2) for idx, row in df.iterrows()]
    
    # Custo "Predito" pela MLP se o nó for atuar como CH na próxima rodada
    df['predicted_ch_cost'] = df['dist_to_gw'].apply(calculate_tx_cost)
    
    fnd, hnd, lnd = -1, -1, -1
    total_nodes = len(df)
    
    round_idx = 0
    alive_count = total_nodes
    
    while alive_count > 0:
        round_idx += 1
        
        df_alive = df[~df['is_dead']].copy()
        if df_alive.empty:
            break
            
        def min_max(s):
            if s.max() == s.min(): return pd.Series(1.0, index=s.index)
            return (s - s.min()) / (s.max() - s.min())
        
        # ========================================================
        # A GRANDE DIFERENÇA DA TESE AQUI
        # ========================================================
        if mode == "predita":
            # IA PREEMPTIVA: O RAMO usa a bateria de agora MENOS o tombo que ele vai levar se for eleito!
            bateria_considerada = df_alive['battery'] - df_alive['predicted_ch_cost']
        else:
            # REACTIVA: O RAMO não enxerga o futuro. Ele avalia só a bateria de agora.
            bateria_considerada = df_alive['battery']
            
        df_alive['norm_energy'] = min_max(bateria_considerada)
        df_alive['norm_prox_wc'] = 1.0 - min_max(df_alive['dist_to_wc'])
        df_alive['norm_prox_gw'] = 1.0 - min_max(df_alive['dist_to_gw'])
        df_alive['norm_density'] = min_max(df_alive['density'])
        
        # Equação RAMO (40% Bateria, 30% GW, 15% WC, 15% Densidade)
        df_alive['score'] = (0.40 * df_alive['norm_energy']) + (0.30 * df_alive['norm_prox_gw']) + (0.15 * df_alive['norm_prox_wc']) + (0.15 * df_alive['norm_density'])
        
        # Elege os Lideres
        ch_ids = []
        for cid in range(k_opt):
            c_nodes = df_alive[df_alive['cluster_id'] == cid]
            if not c_nodes.empty:
                ch_id = c_nodes.loc[c_nodes['score'].idxmax(), 'node_id']
                ch_ids.append(ch_id)
                
        # ========================================================
        # DRENANDO A BATERIA (FIM DA RODADA)
        # ========================================================
        for idx, row in df.iterrows():
            if row['is_dead']: continue
            
            if row['node_id'] in ch_ids:
                # O CH gasta horrores mandando pro Gateway
                cost = calculate_tx_cost(row['dist_to_gw'])
            else:
                # O Nó normal gasta pouquíssimo mandando pro vizinho CH
                cost = calculate_tx_cost(row['dist_to_wc'])
                
            df.at[idx, 'battery'] -= cost
            
            # Se a bateria zerou, ele morre
            if df.at[idx, 'battery'] <= 0:
                df.at[idx, 'battery'] = 0
                df.at[idx, 'is_dead'] = True
                alive_count -= 1
                
                # Registra as métricas oficiais de tese
                if fnd == -1:
                    fnd = round_idx
                if alive_count <= total_nodes / 2 and hnd == -1:
                    hnd = round_idx
                if alive_count == 0:
                    lnd = round_idx
                    
    return fnd, hnd, lnd

print("=======================================================")
print("Iniciando Simulador de Vida Útil (Lifetime) E-DBC")
print("=======================================================\n")

print("[1/2] Executando RAMO Clássico (Reactivo)...")
fnd_res, hnd_res, lnd_res = run_simulation("residual")

print("[2/2] Executando RAMO c/ IA da MLP (Preemptivo)...")
fnd_pre, hnd_pre, lnd_pre = run_simulation("predita")

print("\n=======================================================")
print("          RESULTADOS GLOBAIS DE LIFETIME")
print("=======================================================")
print(f"Métrica               | RAMO Clássico | RAMO + IA MLP")
print(f"----------------------|---------------|----------------")
print(f"First Node Dead (FND) | {fnd_res:13} | {fnd_pre:13}")
print(f"Half Nodes Dead (HND) | {hnd_res:13} | {hnd_pre:13}")
print(f"Last Node Dead  (LND) | {lnd_res:13} | {lnd_pre:13}")
print("=======================================================")

# Gerar gráfico comparativo
labels = ['Primeiro a Morrer\n(FND)', 'Metade da Rede\n(HND)', 'Último a Morrer\n(LND)']
res_vals = [fnd_res, hnd_res, lnd_res]
pre_vals = [fnd_pre, hnd_pre, lnd_pre]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, res_vals, width, label='RAMO S/ IA (Reactivo)', color='#ff4d4d', edgecolor='black')
rects2 = ax.bar(x + width/2, pre_vals, width, label='RAMO C/ IA (Preemptivo)', color='#4da6ff', edgecolor='black')

ax.set_ylabel('Sobrevivência da Rede (Qtd de Transmissões / Rodadas)')
ax.set_title('Impacto da Predição da MLP no Ciclo de Vida da Rede LoRaWAN', pad=20, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Adiciona os números em cima das barras
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), 
                textcoords="offset points",
                ha='center', va='bottom', weight='bold')

plt.grid(axis='y', alpha=0.3)
out_fig = '/home/cristofe/Documents/Project/MLP/recent_logs/lifetime_comparativo.png'
plt.savefig(out_fig, bbox_inches='tight', dpi=300)
print(f"\nGráfico exportado em: {out_fig}")
