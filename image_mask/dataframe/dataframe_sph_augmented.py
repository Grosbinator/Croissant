import pandas as pd
import albumentations as A
from PIL import Image
import numpy as np
import os
from cal_coeff import batch_coefficients2D

# Chemins
csv_path = r"C:\Users\labdsp\Desktop\Croissant_2\Croissant\image_mask\dataset_sphericity.csv"
aug_dir = r"C:\Users\labdsp\Desktop\Croissant_2\Croissant\image_mask\augmented"
os.makedirs(aug_dir, exist_ok=True)

# Charger le CSV
df = pd.read_csv(csv_path)

# Transformations
transforms = [
    ("rot30", A.Rotate(limit=(30, 30), p=1.0)),
    ("rot-30", A.Rotate(limit=(-30, -30), p=1.0)),
    ("flip_h", A.HorizontalFlip(p=1.0)),
    ("flip_v", A.VerticalFlip(p=1.0)),
    ("noise", A.GaussNoise(p=1.0)),
    ("brightness_contrast", A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0)),
    ("gamma", A.RandomGamma(gamma_limit=(80, 120), p=1.0)),
    ("hue_sat", A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0)),
    ("rgb_shift", A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0)),
    ("blur", A.Blur(blur_limit=3, p=1.0)),
    ("grid_dist", A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0)),
    ("clahe", A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0)),
]

new_rows = []

def calc_sphericity(mask_path):
    # Utilise batch_coefficients2D sur un seul masque
    res = batch_coefficients2D(os.path.dirname(mask_path), [os.path.basename(mask_path)])
    if res and "sphericity" in res[0]:
        return res[0]["sphericity"]
    return None

for idx, row in df.iterrows():
    img_path = row['image_path']
    mask_path = row['mask_path']
    label = row['class']

    # Ajoute l'original
    sphericity = row['sphericity'] if 'sphericity' in row else calc_sphericity(mask_path)
    new_rows.append({
        'image_path': img_path,
        'mask_path': mask_path,
        'class': label,
        'sphericity': sphericity
    })

    # Augmentations
    img = np.array(Image.open(img_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L'))

    for suffix, transform in transforms:
        augmented = transform(image=img, mask=mask)
        img_aug = Image.fromarray(augmented['image'])
        mask_aug = Image.fromarray(augmented['mask'])

        base_img = os.path.splitext(os.path.basename(img_path))[0]
        base_mask = os.path.splitext(os.path.basename(mask_path))[0]
        ext_img = os.path.splitext(img_path)[1]
        ext_mask = os.path.splitext(mask_path)[1]

        img_aug_path = os.path.join(aug_dir, f"{base_img}_{suffix}{ext_img}")
        mask_aug_path = os.path.join(aug_dir, f"{base_mask}_{suffix}{ext_mask}")

        img_aug.save(img_aug_path)
        mask_aug.save(mask_aug_path)

        # Calcul de la sphericity sur le mask transformé
        sph = calc_sphericity(mask_aug_path)

        new_rows.append({
            'image_path': img_aug_path,
            'mask_path': mask_aug_path,
            'class': label,
            'sphericity': sph
        })

# Fusionner et sauvegarder le nouveau CSV
df_aug = pd.DataFrame(new_rows)
df_aug.to_csv(r"C:\Users\labdsp\Desktop\Croissant_2\Croissant\image_mask\dataset_sphericity_augmented.csv", index=False)