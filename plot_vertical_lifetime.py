import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.3

# 1. RUN LIFETIME SIMULATION
df_init = pd.read_csv('/home/cristofe/Documents/Project/MLP/recent_logs/node_with_clusters.csv')
k_opt = df_init['cluster_id'].nunique()
weak_centroids = []
for cid in range(k_opt):
    c_nodes = df_init[df_init['cluster_id'] == cid]
    weak_centroids.append([c_nodes['x'].mean(), c_nodes['y'].mean()])
weak_centroids = np.array(weak_centroids)

gw_x, gw_y = 0, 0
df_init['dist_to_wc'] = [np.sqrt((row['x'] - weak_centroids[int(row['cluster_id'])][0])**2 + (row['y'] - weak_centroids[int(row['cluster_id'])][1])**2) for idx, row in df_init.iterrows()]
df_init['dist_to_gw'] = [np.sqrt((row['x'] - gw_x)**2 + (row['y'] - gw_y)**2) for idx, row in df_init.iterrows()]

def calculate_tx_cost(distance):
    return 0.0005 + (0.000005 * distance)
df_init['predicted_ch_cost'] = df_init['dist_to_gw'].apply(calculate_tx_cost)

survival_curves = {}
for mode in ['residual', 'predita']:
    df = df_init.copy()
    df['battery'] = 1.0
    df['is_dead'] = False
    
    rounds_alive = [len(df)]
    alive_count = len(df)
    while alive_count > 0:
        df_alive = df[~df['is_dead']].copy()
        if df_alive.empty: break
        
        def min_max(s):
            if s.max() == s.min(): return pd.Series(1.0, index=s.index)
            return (s - s.min()) / (s.max() - s.min())
        
        if mode == 'predita':
            bateria_considerada = df_alive['battery'] - df_alive['predicted_ch_cost']
        else:
            bateria_considerada = df_alive['battery']
            
        df_alive['norm_energy'] = min_max(bateria_considerada)
        df_alive['norm_prox_wc'] = 1.0 - min_max(df_alive['dist_to_wc'])
        df_alive['norm_prox_gw'] = 1.0 - min_max(df_alive['dist_to_gw'])
        df_alive['norm_density'] = min_max(df_alive['density'])
        df_alive['score'] = (0.40 * df_alive['norm_energy']) + (0.30 * df_alive['norm_prox_gw']) + (0.15 * df_alive['norm_prox_wc']) + (0.15 * df_alive['norm_density'])
        
        ch_ids = set()
        for cid in range(k_opt):
            c_nodes = df_alive[df_alive['cluster_id'] == cid]
            if not c_nodes.empty:
                ch_id = c_nodes.loc[c_nodes['score'].idxmax(), 'node_id']
                ch_ids.add(ch_id)
                
        is_alive_mask = ~df['is_dead']
        is_ch_mask = is_alive_mask & df['node_id'].isin(ch_ids)
        is_cm_mask = is_alive_mask & ~df['node_id'].isin(ch_ids)
        
        cost_ch = calculate_tx_cost(df.loc[is_ch_mask, 'dist_to_gw'])
        cost_cm = calculate_tx_cost(df.loc[is_cm_mask, 'dist_to_wc'])
        
        df.loc[is_ch_mask, 'battery'] -= cost_ch
        df.loc[is_cm_mask, 'battery'] -= cost_cm
        
        dead_now = df['battery'] <= 0
        df.loc[dead_now, 'battery'] = 0
        df.loc[dead_now, 'is_dead'] = True
        alive_count = (~df['is_dead']).sum()
        rounds_alive.append(alive_count)
        
    survival_curves[mode] = rounds_alive

# 2. LOAD LORASIMPLUS TRACES
l_rec = pd.read_csv('/home/cristofe/LoRaSimPlus/results/2026-04-10_16-15-27/links.csv')
l_pre = pd.read_csv('/home/cristofe/LoRaSimPlus/results/2026-07-31_21-06-00/links.csv')

e_rec = l_rec.groupby('node_id')['energy_j'].sum() * 1000
e_pre = l_pre.groupby('node_id')['energy_j'].sum() * 1000
e_rec = e_rec.reindex(range(500), fill_value=0.0).values
e_pre = e_pre.reindex(range(500), fill_value=0.0).values

# 3. VERTICAL FIGURE (2 ROWS, 1 COLUMN - IDEAL FOR SINGLE COLUMN)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 9.5))

col_reactive = '#E74C3C'
col_predictive = '#2980B9'
rounds_x_rec = np.arange(len(survival_curves['residual']))
rounds_x_pre = np.arange(len(survival_curves['predita']))

# 1. Base solid line for Predictive RAMO with blue square markers
ax1.plot(rounds_x_pre, survival_curves['predita'], label='Predictive RAMO (MLP)',
         color='#1F618D', linewidth=2.4, zorder=2,
         marker='s', markevery=(35, 70), markersize=6.0,
         markerfacecolor='#1F618D', markeredgecolor='black')

# 2. Overlay dashed vibrant red line for Reactive RAMO with red circle markers
ax1.plot(rounds_x_rec, survival_curves['residual'], label='Reactive RAMO',
         color='#E74C3C', linewidth=2.8, linestyle='--', dashes=(8, 5), zorder=3,
         marker='o', markevery=(0, 70), markersize=6.5,
         markerfacecolor='#E74C3C', markeredgecolor='black')

ax1.axvline(x=81, color='#555555', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.axvline(x=706, color='#555555', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.axvline(x=1152, color='#555555', linestyle=':', linewidth=1.5, alpha=0.7)

ax1.annotate('FND (Round 81)', xy=(81, 480), xytext=(120, 485),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
             fontsize=12, fontweight='bold', color='#222222',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#F9F9F9', edgecolor='#CCCCCC'))

ax1.annotate('HND (Round 706)', xy=(706, 250), xytext=(740, 270),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
             fontsize=12, fontweight='bold', color='#222222',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#F9F9F9', edgecolor='#CCCCCC'))

ax1.annotate('LND (Round 1152)', xy=(1152, 10), xytext=(850, 70),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
             fontsize=12, fontweight='bold', color='#222222',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#F9F9F9', edgecolor='#CCCCCC'))

ax1.set_title('(a) Network Survival Trajectory Across Rounds', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('Simulation Rounds', fontsize=13, fontweight='bold')
ax1.set_ylabel('Active Nodes (Count)', fontsize=13, fontweight='bold')
ax1.set_ylim(-20, 530)
ax1.set_xlim(-10, 1200)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(loc='lower left', fontsize=12, framealpha=0.95, edgecolor='#BBBBBB')
ax1.grid(True, linestyle='--', alpha=0.4)

# Subplot (b)
box_data = [e_rec, e_pre]
bp = ax2.boxplot(box_data, patch_artist=True, widths=0.45,
                 medianprops=dict(color='black', linewidth=2.0),
                 whiskerprops=dict(linewidth=1.5, color='#444444'),
                 capprops=dict(linewidth=1.5, color='#444444'),
                 flierprops=dict(marker='o', markersize=4, alpha=0.35, markeredgecolor='none'))

bp['boxes'][0].set(facecolor=col_reactive, alpha=0.7, edgecolor='black', linewidth=1.5)
bp['boxes'][1].set(facecolor=col_predictive, alpha=0.7, edgecolor='black', linewidth=1.5)

means = [np.mean(e_rec), np.mean(e_pre)]
ax2.scatter([1, 2], means, color='gold', edgecolor='black', s=120, zorder=5, marker='D', label='Mean')

mean_diff = ((means[1] - means[0]) / means[0]) * 100
max_diff = ((np.max(e_pre) - np.max(e_rec)) / np.max(e_rec)) * 100

ax2.text(1, means[0] + 3.0, f"Mean: {means[0]:.2f} mJ\nMax: {np.max(e_rec):.1f} mJ",
         ha='center', va='bottom', fontsize=11, fontweight='bold', color='#900C3F',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#E74C3C', alpha=0.85))

ax2.text(2, means[1] + 3.0, f"Mean: {means[1]:.2f} mJ\nMax: {np.max(e_pre):.1f} mJ",
         ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1B4F72',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#2980B9', alpha=0.85))

efficiency_summary = (
    "Predictive RAMO Benefits:\n"
    f"• Mean Node Energy: {mean_diff:.1f}%\n"
    f"• Peak Node Energy: {max_diff:.1f}%\n"
    "• Energy / Received Packet: -26.1%"
)
ax2.text(0.5, 0.95, efficiency_summary, transform=ax2.transAxes,
         fontsize=11.5, fontweight='bold', va='top', ha='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#EAFAF1', edgecolor='#27AE60', linewidth=1.4))

ax2.set_title('(b) Individual Node Energy Consumption', fontsize=14, fontweight='bold', pad=10)
ax2.set_xticklabels(['Reactive\nRAMO', 'Predictive\nRAMO (MLP)'], fontsize=12, fontweight='bold')
ax2.set_ylabel('Energy Consumed per Node (mJ)', fontsize=13, fontweight='bold')
ax2.set_ylim(-2, 105)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax2.grid(True, linestyle='--', alpha=0.4, axis='y')

plt.tight_layout(pad=2.2)

out_vertical = '/home/cristofe/Documents/Project/MLP/figuras/lifetime_vertical.png'
plt.savefig(out_vertical, dpi=300, bbox_inches='tight')
print(f"Vertical figure generated at: {out_vertical}")
