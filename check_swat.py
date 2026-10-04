import pandas as pd

normal_path = "data/swat/SWaT_Dataset_Normal_v1.xlsx"
attack_path = "data/swat/SWaT_Dataset_Attack_v0.xlsx"

normal = pd.read_excel(normal_path, header=1)
attack = pd.read_excel(attack_path, header=1)

print("\nNORMAL DATA")
print(normal.shape)
print(normal.head())

print("\nATTACK DATA")
print(attack.shape)
print(attack.head())

print("\nNORMAL COLUMNS")
print(normal.columns.tolist())

print("\nATTACK COLUMNS")
print(attack.columns.tolist())

print("\nATTACK LABEL COUNTS")
print(attack["Normal/Attack"].value_counts())