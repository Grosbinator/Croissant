import numpy
import os
import SimpleITK as sitk
import numpy as np

def calculate_coefficients2D(maskArray, pixelSpacing):
    import cv2
    mask = (maskArray > 0).astype(numpy.uint8)
    area = numpy.sum(mask) * numpy.prod(pixelSpacing)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 0
    perimeter = cv2.arcLength(contours[0], True) * pixelSpacing[0]
    # Pour le diamètre maximal, on peut utiliser la distance max entre deux points du contour
    pts = contours[0][:,0,:]
    max_diam = 0
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = numpy.linalg.norm((pts[i] - pts[j]) * pixelSpacing)
            if d > max_diam:
                max_diam = d
    return perimeter, area, max_diam

def batch_coefficients2D(mask_dir):
    """
    Applique calculate_coefficients2D à tous les masques d'un dossier.
    Retourne une liste de dictionnaires avec le chemin, le périmètre, la surface, le diamètre,
    et la sphéricité calculée à partir du périmètre et de la surface.
    """
 

    results = []
    for fname in os.listdir(mask_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            continue
        mask_path = os.path.join(mask_dir, fname)
        mask_img = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask_img)
        # Si le masque est 3D, prendre le premier plan
        if mask_array.ndim == 3:
            mask_array = mask_array[0]
        spacing = mask_img.GetSpacing()[:2]
        perimeter, surface, diameter = calculate_coefficients2D(mask_array, spacing)
        # Calcul de la sphéricité (évite division par zéro)
        if perimeter == 0:
            sphericity = np.nan
        else:
            sphericity = (2 * np.sqrt(np.pi * surface)) / perimeter
        results.append({
            "mask_path": mask_path,
            "perimeter": perimeter,
            "surface": surface,
            "diameter": diameter,
            "sphericity": sphericity
        })
    return results