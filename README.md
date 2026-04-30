# Soli Gesture Recognition - ELL784 Assignment 3

Adversarial training for hand gesture recognition on Google Soli radar.

**Method:** cWGAN-GP (data augmentation) + DANN with Gradient Reversal Layer (cross-subject generalization)

**Results:** 92.86% mean accuracy, 85.80% fine-grained accuracy (2-fold cross-validation)

## Repository Contents

- Soli_Final_ML.ipynb - Complete training and evaluation notebook (Google Colab)
- README.md - This file

## Dataset

Download from: https://github.com/simonwsw/deep-soli

Click the dataset link in their README. Extract zip to get dsp/ folder with 5500 .h5 files.

## How to Run (Google Colab)

1. Open Soli_Final_ML.ipynb in Google Colab
2. Enable GPU: Runtime > Change runtime type > T4 GPU
3. Upload SoliData.zip using the folder icon on the left
4. Run all cells from top to bottom

The notebook automatically installs dependencies, extracts data, trains the GAN, trains the DANN classifier, evaluates results, and downloads model checkpoints.

## Pre-trained Model Weights

Download from Google Drive:

- best_model_fold0.pt : https://drive.google.com/YOUR_LINK_HERE
- best_model_fold1.pt : https://drive.google.com/YOUR_LINK_HERE
- generator_final.pt  : https://drive.google.com/YOUR_LINK_HERE

## Results

Two-fold cross-validation over subjects:

| | Fold 1 | Fold 2 | Mean |
|---|---|---|---|
| Overall Accuracy | 90.61% | 95.12% | 92.86% |
| Fine-Grained Accuracy | 78.17% | 93.43% | 85.80% |

Per-class accuracy:

| Class | Fold 1 | Fold 2 | Mean |
|---|---|---|---|
| pinch_index (FG) | 86.00% | 89.14% | 87.57% |
| pinch_pinky (FG) | 97.00% | 98.29% | 97.64% |
| finger_slide (FG) | 76.00% | 94.29% | 85.14% |
| finger_rub (FG) | 53.67% | 92.00% | 72.83% |
| slow_swipe | 96.33% | 96.57% | 96.45% |
| fast_swipe | 99.33% | 96.00% | 97.67% |
| push | 99.33% | 98.29% | 98.81% |
| pull | 98.00% | 98.29% | 98.14% |
| palm_tilt | 99.33% | 98.86% | 99.10% |
| circle | 95.33% | 96.57% | 95.95% |
| palm_hold | 96.33% | 88.00% | 92.17% |

Ablation study:

| Method | Overall | Fine-Grained |
|---|---|---|
| Baseline (3D-CNN only) | 90.79% | 85.87% |
| + GAN augmentation | 87.89% | 84.16% |
| + DANN only | 84.94% | 73.46% |
| Full (GAN + DANN) | 91.69% | 85.57% |

## Architecture

- Encoder: 3D-CNN -> 256-dim feature vector
- Gesture Classifier: Linear layers -> 11 classes
- Domain Discriminator: GRL + Linear layers -> 10 subjects
- Generator: Noise + class label -> fake radar clip (32x32x40)

## Hyperparameters

- GAN epochs: 50
- DANN epochs: 30
- Synthetic samples per class: 200
- Domain loss weight: 0.1
- GAN learning rate: 2e-4
- DANN learning rate: 1e-3
- Batch size: 32

## References

1. Wang et al. Interacting with Soli. ACM UIST 2016.
2. Ganin et al. Domain-Adversarial Training of Neural Networks. JMLR 2016.
3. Gulrajani et al. Improved Training of Wasserstein GANs. NeurIPS 2017.
