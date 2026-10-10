import os
import shutil
import glob

results_dir = "/home/cristofe/LoRaSimPlus/results"
target_dir = "/home/cristofe/Documents/Project/MLP/recent_logs"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

all_dirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
# Filter only those that look like a date (e.g. 2026-04-10...)
valid_dirs = [d for d in all_dirs if '2026' in d]
valid_dirs.sort(key=os.path.getmtime, reverse=True)

for run_dir in valid_dirs:
    csv_files = glob.glob(f"{run_dir}/*.csv")
    txt_files = glob.glob(f"{run_dir}/*.txt")
    if csv_files or txt_files:
        print(f"Pasta encontrada com dados: {run_dir}")
        for f in csv_files + txt_files:
            print(f"Copiando {os.path.basename(f)} para {target_dir}...")
            shutil.copy(f, target_dir)
        break
    else:
        print(f"Ignorando pasta vazia: {run_dir}")

print("Pronto!")
