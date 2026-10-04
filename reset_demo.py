import os

os.makedirs("data", exist_ok=True)
with open("data/attack_mode.txt", "w", encoding="utf-8") as f:
    f.write("normal")
print("Demo reset to normal operation.")
