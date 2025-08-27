
# Croissant

Breast cancer image classification using PyTorch Lightning.

This project focuses on classifying breast ultrasound images to detect cancer. The dataset used is the [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) from Kaggle.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python only_pic_lightning/train/train_lightning_D_CE.py
```


## Project Structure

- `only_pic/` : Initial experiments and learning scripts (basic version)
	- `dataframe.py` : Dataset DataFrame creation/management.
	- `only_pic_transform.csv`, `only_pic.csv` : CSV files with dataset annotations or splits.
	- `train/`
		- `dataloader.py` : DataLoader creation for images and labels.
		- `loss.py` : Custom loss functions.
		- `model_lightning.py` : PyTorch Lightning model definitions (binary and multiclass).
		- `models.py` : Neural network architectures (ResNet, MobileNet, etc.).
		- `train_lightning.py`, `train.py` : Training scripts.
- `only_pic_lightning/` : More developed and modular Lightning version (recommended)
	- `dataframe_transform.py`, `dataframe.py` : Advanced DataFrame and transformation management.
	- `only_pic_transform_balanced.csv`, `only_pic_transform.csv`, `only_pic.csv` : CSV files for dataset splits and balancing.
	- `train/`
		- `confmat_custom.png` : Generated confusion matrix image.
		- `confusion_mat rix.py` : Confusion matrix generation and display.
		- `dataloader.py` : DataLoaders for the Lightning version.
		- `loss.py` : Loss functions for Lightning.
		- `model_lightning.py` : Lightning models (multiclass, patient-level, etc.).
		- `models.py` : Network architectures for Lightning.
		- `train_light_D_CE.py` : Lightning training scripts.
		- `try.ipynb` : Experimentation notebook.
- `README.md` : This file
- `requirements.txt` : Python dependencies

## Author

Grosbinator
