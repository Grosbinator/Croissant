
# Croissant

Breast cancer ultrasound image classification using PyTorch Lightning.

This project aims to classify breast ultrasound images to detect cancer, using various deep learning models and a modular structure based on PyTorch Lightning. The main dataset is the [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) from Kaggle.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Example to train the main model (DenseNet + CrossEntropy):
```bash
python only_pic_lightning/train/train_lightning_D_CE.py
```

## Project Structure

- `only_pic/` : Initial experiments and basic scripts
	- `dataframe.py` : Dataset DataFrame creation and management
	- `train/` :
		- `dataloader.py` : DataLoader for images and labels
		- `train.py`, `train_lightning.py` : Training scripts
- `only_pic_lightning/` : More modular and recommended Lightning version
	- `dataframe.py`, `dataframe_transform.py` : Advanced DataFrame and transformation management
	- `train/` :
		- `train_light_D_CE.py` : Main training script (DenseNet + CrossEntropy)
		- `model_lightning.py` : Lightning model definitions (binary, multiclass, patient-level)
		- `models.py` : Network architectures (ResNet, MobileNet, DenseNet, etc.)
		- `loss.py` : Custom loss functions
		- `dataloader.py` : DataLoaders for Lightning
		- `confusion_matrix.py` : Confusion matrix generation and display

- `image_mask/` : **Modules for image segmentation and masking**
	- `dataframe/` : DataFrame scripts for segmentation
		- [`cal_coeff.py`](image_mask/dataframe/cal_coeff.py) : Calculates calibration coefficients for image normalization.
		- [`dataframe_sph.py`](image_mask/dataframe/dataframe_sph.py) : Builds DataFrames for spherical mask datasets.
		- [`dataframe_sph_augmented.py`](image_mask/dataframe/dataframe_sph_augmented.py) : DataFrame creation with data augmentation for spherical masks.
	- `dataset/` : Data organization notebooks
		- [`organizing.ipynb`](image_mask/dataset/organizing.ipynb) : Jupyter notebook for organizing and preprocessing segmentation datasets.
	- `model/` : Segmentation models and training scripts
		- [`dataloader.py`](image_mask/model/dataloader.py) : DataLoader for segmentation images and masks.
		- [`LateFusionLightning.py`](image_mask/model/LateFusionLightning.py) : PyTorch Lightning model for late fusion segmentation.
		- [`LF_models.py`](image_mask/model/LF_models.py) : Contains late fusion model architectures.
		- [`loss.py`](image_mask/model/loss.py) : Custom loss functions for segmentation tasks.
		- [`only_pic.py`](image_mask/model/only_pic.py) : Segmentation script using only image data.
		- [`oui.ipynb`](image_mask/model/oui.ipynb) : Notebook for segmentation experiments and visualization.
		- [`train.py`](image_mask/model/train.py) : Main training script for segmentation models.
		- [`train2.py`](image_mask/model/train2.py) : Alternative training script for segmentation.
		- [`train_3.py`](image_mask/model/train_3.py) : Third version of segmentation training script. You can forget the others this one working fine and is easy to use.
- `General/` : Attempt to gather all models and losses in one place. (Note: Due to lack of time, this module is not fully(at all(sry Gab)) functional.)
- `pyra/` : Alternative experimental scripts
- `requirements.txt` : Python dependencies

## Main Features

- Binary and multiclass classification
- Advanced metrics (accuracy, recall, specificity, confusion matrix)
- Early stopping and learning rate scheduler
- Modular architectures and loss functions

## Author

Grosbinator
