# SafeByte

SafeByte is an Autoencoder-based EXE analyzer that helps detect anomalies in executable files. It trains on clean (benign) EXEs and can identify suspicious files with confidence scores.

## Features

- Convert EXE files to images for training.
- Data augmentation to increase training diversity.
- Train an Autoencoder on clean files only.
- Detect suspicious EXEs using reconstruction error.
- Confidence score estimation for each detection.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SafeByte.git
   cd SafeByte
Install dependencies:

bash
نسخ
تحرير
pip install -r requirements.txt
Usage
Training
Train the autoencoder on your clean EXE dataset:

bash
نسخ
تحرير
python train_autoencoder_clean_only.py --train
Detection
Detect a new EXE file using the trained model:

bash
نسخ
تحرير
python train_autoencoder_clean_only.py --detect path/to/file.exe
Configuration
CLEAN_EXE_DIR: Directory containing clean EXEs for training.

IMG_SIZE: Size of images generated from EXEs (default: 128x128).

BATCH_SIZE: Training batch size.

EPOCHS: Number of training epochs.

AUG_IMG_DIR: Directory to store augmented images.

MODEL_PATH: Path to save the trained Autoencoder model.

THRESHOLD_PATH: Path to save the reconstruction error threshold.

Notes
Currently designed to train on benign files only.

Reconstruction error threshold is automatically calculated as mean + 3 * std.

Confidence score is estimated between 0 (low) and 1 (high).
