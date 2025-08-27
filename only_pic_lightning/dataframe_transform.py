import pandas as pd
import albumentations as A
from PIL import Image
import numpy as np
import os

# Chemins
csv_path = r"C:\Users\labdsp\Documents\Croissant\only_pic\only_pic.csv"
aug_dir = r"C:\Users\labdsp\Documents\Croissant\only_pic\augmented"
os.makedirs(aug_dir, exist_ok=True)

# Charger le CSV
df = pd.read_csv(csv_path)

# Séparer les classes
df_0 = df[df['class'] == 0]
df_1 = df[df['class'] == 1]

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

def augment_and_balance(df_class, class_label, n_target):
    images = df_class['image_path'].tolist()
    n_orig = len(images)
    # Ajoute l'original
    for img_path in images:
        new_rows.append({'image_path': img_path, 'class': class_label})
    # Applique les augmentations en boucle jusqu'à n_target
    i = 0
    while len([r for r in new_rows if r['class'] == class_label]) < n_target:
        img_path = images[i % n_orig]
        img = np.array(Image.open(img_path).convert('RGB'))
        suffix, transform = transforms[i % len(transforms)]
        augmented = transform(image=img)
        img_aug = Image.fromarray(augmented['image'])
        base = os.path.basename(img_path)
        new_path = os.path.join(aug_dir, f"{class_label}_{suffix}_{i}_{base}")
        img_aug.save(new_path)
        new_rows.append({'image_path': new_path, 'class': class_label})
        i += 1

# Choisis le nombre cible (par exemple, le double du max initial)
n_max = max(len(df_0), len(df_1))
n_target = n_max * 2

augment_and_balance(df_0, 0, n_target)
augment_and_balance(df_1, 1, n_target)

# Fusionner et sauvegarder le nouveau CSV
df_aug = pd.DataFrame(new_rows)
df_aug.to_csv(r"C:\Users\labdsp\Documents\Croissant\only_pic_lightning\only_pic_transform_balanced.csv", index=False)

# 2 transformations : flip horizontal et rotation de 30° 
# import pandas as pd
# from PIL import Image
# import os

# # Chemins
# csv_path = r"C:\Users\labdsp\Documents\Croissant\only_pic\only_pic.csv"
# flip_dir = r"C:\Users\labdsp\Documents\Croissant\only_pic\flipped"
# rot_dir = r"C:\Users\labdsp\Documents\Croissant\only_pic\rotated"
# os.makedirs(flip_dir, exist_ok=True)
# os.makedirs(rot_dir, exist_ok=True)

# # 1. Charger le CSV
# df = pd.read_csv(csv_path)

# # 2. Filtrer les images de la classe 1
# df_class1 = df[df['class'] == 1]

# new_rows = []

# # 3. Flip horizontal
# for idx, row in df_class1.iterrows():
#     img = Image.open(row['image_path']).convert('RGB')
#     img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
#     base = os.path.basename(row['image_path'])
#     new_path = os.path.join(flip_dir, f"flip_{base}")
#     img_flipped.save(new_path)
#     new_rows.append({'image_path': new_path, 'class': 1})

# # 4. Rotation de 30°
# for idx, row in df_class1.iterrows():
#     img = Image.open(row['image_path']).convert('RGB')
#     img_rot = img.rotate(30)
#     base = os.path.basename(row['image_path'])
#     new_path = os.path.join(rot_dir, f"rot_{base}")
#     img_rot.save(new_path)
#     new_rows.append({'image_path': new_path, 'class': 1})

# # 5. Fusionner et sauvegarder le nouveau CSV
# df_aug = pd.DataFrame(new_rows)
# df_new = pd.concat([df, df_aug], ignore_index=True)
# df_new.to_csv(r"C:\Users\labdsp\Documents\Croissant\only_pic\only_pic_transform.csv", index=False)