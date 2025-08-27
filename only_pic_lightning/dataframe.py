import os
import pandas as pd

base_dir = r"C:\Users\labdsp\Documents\Croissant\only_pic\dataset"

dirs = {
    "benign": {
        "images": os.path.join(base_dir, "benign_images"),
        "label": 0
    },
    "malign": {
        "images": os.path.join(base_dir, "malign_images"),
        "label": 1
    }
}

records = []

for cls, paths in dirs.items():
    img_dir  = paths["images"]
    label    = paths["label"]

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            continue

        img_path  = os.path.join(img_dir, fname)
        name, ext = os.path.splitext(fname)
 
        records.append({
            "image_path":  img_path,
            "class":       label,        
            })

df = pd.DataFrame(records)
print(df)
df.to_csv("only_pic.csv", index=False)