import os
import pandas as pd
from cal_coeff import batch_coefficients2D

base_dir = r"C:\Users\labdsp\Desktop\Croissant_2\Croissant\image_mask\dataset"

dirs = {
    "benign": {
        "images": os.path.join(base_dir, "benign_images"),
        "masks":  os.path.join(base_dir, "benign_mask"),
        "label": 0
    },
    "malign": {
        "images": os.path.join(base_dir, "malign_images"),
        "masks":  os.path.join(base_dir, "malign_mask"),
        "label": 1
    }
}

records = []

for cls, paths in dirs.items():
    img_dir  = paths["images"]
    mask_dir = paths["masks"]
    label    = paths["label"]

    # Utilise batch_coefficients2D pour tous les masques du dossier
    mask_results = batch_coefficients2D(mask_dir)
    # Crée un dictionnaire pour retrouver rapidement les résultats par nom de masque
    mask_dict = {os.path.basename(r["mask_path"]): r for r in mask_results}

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            continue

        img_path  = os.path.join(img_dir, fname)
        name, ext = os.path.splitext(fname)
        mask_fname = f"{name}_mask{ext}"
        mask_path = os.path.join(mask_dir, mask_fname)

        if not os.path.exists(mask_path):
            continue

        # Cherche les coefficients du masque correspondant
        mask_info = mask_dict.get(os.path.basename(mask_path))
        if mask_info is None:
            continue

        records.append({
            "image_path":  img_path,
            "mask_path":   mask_path,
            "class":       label,
            "sphericity":  mask_info["sphericity"]
        })

df = pd.DataFrame(records)
print(df)
df.to_csv("dataset_sphericity.csv", index=False)