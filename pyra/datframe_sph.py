import pandas as pd
import albumentations as A
from PIL import Image
import numpy as np
import os
import SimpleITK as sitk
from pyra import RadiomicsShape2D
import cv2

# Chemins
csv_path = r"C:\Users\labdsp\Desktop\Croissant_2\Croissant\only_pic_lightning\only_pic.csv"
aug_dir = r"C:\Users\labdsp\Desktop\Croissant_2\Croissant\only_pic_lightning\augmented"
os.makedirs(aug_dir, exist_ok=True)

# Charger le CSV (doit contenir 'image_path' et 'mask_path')
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

def pil_to_sitk(img):
    """Convertit une image PIL en SimpleITK (uint8)."""
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[..., 0]  # Prend le canal rouge si RGB
    return sitk.GetImageFromArray(arr.astype(np.uint8))

for idx, row in df.iterrows():
    img_path = row['image_path']
    mask_path = row['mask_path']
    label = row['class']

    # Ajoute l'original
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    img_save_path = os.path.join(aug_dir, f"orig_{os.path.basename(img_path)}")
    mask_save_path = os.path.join(aug_dir, f"orig_{os.path.basename(mask_path)}")
    img.save(img_save_path)
    mask.save(mask_save_path)

    # Calcul de la sphericity sur l'original
    itk_img = pil_to_sitk(img)
    itk_mask = pil_to_sitk(mask)
    shape_feat = RadiomicsShape2D(itk_img, itk_mask)
    shape_feat._initSegmentBasedCalculation()
    sphericity = shape_feat.getSphericityFeatureValue()

    new_rows.append({
        'image_path': img_save_path,
        'mask_path': mask_save_path,
        'class': label,
        'sphericity': sphericity
    })

    # Augmentations
    for suffix, transform in transforms:
        img_np = np.array(img)
        mask_np = np.array(mask)
        augmented = transform(image=img_np, mask=mask_np)
        img_aug = Image.fromarray(augmented['image'])
        mask_aug = Image.fromarray(augmented['mask'])

        img_aug_path = os.path.join(aug_dir, f"{suffix}_{idx}_{os.path.basename(img_path)}")
        mask_aug_path = os.path.join(aug_dir, f"{suffix}_{idx}_{os.path.basename(mask_path)}")
        img_aug.save(img_aug_path)
        mask_aug.save(mask_aug_path)

        # Calcul de la sphericity sur l'augmentation
        itk_img_aug = pil_to_sitk(img_aug)
        itk_mask_aug = pil_to_sitk(mask_aug)
        shape_feat_aug = RadiomicsShape2D(itk_img_aug, itk_mask_aug)
        shape_feat_aug._initSegmentBasedCalculation()
        sphericity_aug = shape_feat_aug.getSphericityFeatureValue()

        new_rows.append({
            'image_path': img_aug_path,
            'mask_path': mask_aug_path,
            'class': label,
            'sphericity': sphericity_aug
        })

# Création du DataFrame final
df_aug = pd.DataFrame(new_rows)
df_aug.to_csv(r"C:\Users\labdsp\Desktop\Croissant_2\Croissant\image_mask\transform_sphericity.csv", index=False)