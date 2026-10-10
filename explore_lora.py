import subprocess

lora_dir = "/home/cristofe/LoRaSimPlus"

# List files
print("Arquivos no diretorio do LoRaSimPlus:")
print(subprocess.getoutput(f"ls -l {lora_dir}"))

print("\n\nAjuda do main.py:")
print(subprocess.getoutput(f"python3 {lora_dir}/main.py --help"))
