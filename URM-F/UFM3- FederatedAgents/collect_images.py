import os
import shutil

# Path to project root
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define final destination folders
pneumonia_output = os.path.join(base_dir, "all_pneumonia")
normal_output = os.path.join(base_dir, "all_normal")

# Make sure output dirs exist
os.makedirs(pneumonia_output, exist_ok=True)
os.makedirs(normal_output, exist_ok=True)

# Traverse the entire folder tree
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith((".jpeg", ".jpg", ".png")):
            src_path = os.path.join(root, file)

            # Check if the path contains class name
            if "PNEUMONIA" in root.upper():
                dst_path = os.path.join(pneumonia_output, file)
            elif "NORMAL" in root.upper():
                dst_path = os.path.join(normal_output, file)
            else:
                continue  # ignore anything not pneumonia/normal

            # Avoid overwriting by renaming duplicates
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(file)
                i = 1
                while os.path.exists(os.path.join(os.path.dirname(dst_path), f"{name}_{i}{ext}")):
                    i += 1
                dst_path = os.path.join(os.path.dirname(dst_path), f"{name}_{i}{ext}")

            shutil.copy2(src_path, dst_path)

print("✅ Done! All NORMAL and PNEUMONIA images collected.")
