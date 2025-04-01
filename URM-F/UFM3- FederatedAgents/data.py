import os
import shutil
import random

# Define source folders
source_normal = '/Users/sreebhargavibalija/Desktop/ufl all files/ufm-federated-agent/all_normal'
source_pneumonia = '/Users/sreebhargavibalija/Desktop/ufl all files/ufm-federated-agent/all_pneumonia'

# Create client folders
client_base = 'chestx-ray/clients'
os.makedirs(client_base, exist_ok=True)

for i in range(1, 4):
    os.makedirs(f'{client_base}/client{i}/normal', exist_ok=True)
    os.makedirs(f'{client_base}/client{i}/pneumonia', exist_ok=True)

# Helper function to split and distribute files
def distribute_files(src_dir, label):
    files = os.listdir(src_dir)
    random.shuffle(files)
    split_size = len(files) // 3

    for i in range(3):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < 2 else len(files)  # last client gets the rest
        for file in files[start_idx:end_idx]:
            src_path = os.path.join(src_dir, file)
            dest_path = os.path.join(client_base, f'client{i+1}', label, file)
            shutil.copy(src_path, dest_path)

# Distribute both normal and pneumonia files
distribute_files(source_normal, 'normal')
distribute_files(source_pneumonia, 'pneumonia')
