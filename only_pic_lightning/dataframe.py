import os
import pandas as pd

base_dir = r"C:\Users\labdsp\Desktop\Croissant_2\Croissant\Dataset_BUSI_with_GT"

dirs = {
    "benign": {
        "images": os.path.join(base_dir, "benign"),
        "label": 0
    },
    "malignant": {
        "images": os.path.join(base_dir, "malignant"),
        "label": 1
    }
}

records = []

for cls, paths in dirs.items():
    img_dir  = paths["images"]
    label    = paths["label"]

    # Séparer images et masques
    all_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))]
    images = [f for f in all_files if "_mask" not in f]
    masks = {f.replace(".png", "").replace(".jpg", "").replace(".jpeg", "").replace(".tif", "").replace(".bmp", ""): f for f in all_files if "_mask" in f}

    for img_fname in images:
        name, ext = os.path.splitext(img_fname)
        # Chercher le masque correspondant
        mask_key = name + "_mask"
        mask_fname = None
        for ext_mask in [".png", ".jpg", ".jpeg", ".tif", ".bmp"]:
            if mask_key in masks:
                mask_fname = masks[mask_key]
                break
            elif mask_key + "_1" in masks:  # gestion des masques alternatifs
                mask_fname = masks[mask_key + "_1"]
                break

        img_path = os.path.join(img_dir, img_fname)
        mask_path = os.path.join(img_dir, mask_fname) if mask_fname else None

        records.append({
            "image_path": img_path,
            "mask_path": mask_path,
            "class": label
        })

df = pd.DataFrame(records)
print(df)
df.to_csv("only_pic.csv", index=False)