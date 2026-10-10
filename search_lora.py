import os

search_dir = "/home/cristofe"
target = "lorasimplus"

found = []
for root, dirs, files in os.walk(search_dir):
    # Ignore hidden dirs to speed up
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    
    if target.lower() in root.lower():
        found.append(root)
    for f in files:
        if target.lower() in f.lower():
            found.append(os.path.join(root, f))

print("Found:")
for p in set(found):
    print(p)
