Soli Gesture Recognition — ELL784 Assignment 3
Adversarial training framework for hand gesture recognition on Google Soli radar data. Addresses two problems: fine-grained gesture accuracy and cross-subject generalization.
Method: Conditional WGAN-GP for data augmentation + Domain-Adversarial Neural Network (DANN) with Gradient Reversal Layer.
Results: 92.86% mean overall accuracy and 85.80% mean fine-grained accuracy under two-fold cross-validation over subjects, vs. 79% baseline from the original paper.

Repository Contents

Soli_Final_ML.ipynb — Full training and evaluation notebook (Google Colab)
README.md — This file


Dataset
Download the Soli dataset from: https://github.com/simonwsw/deep-soli
Click the dataset link in their README. Extract the zip to get a dsp/ folder with 5500 .h5 files.
File naming convention: gestureID_sessionID_instanceID.h5

How to Run on Google Colab (Recommended)
Step 1 — Go to colab.research.google.com → File → Upload notebook → select Soli_Final_ML.ipynb
Step 2 — Enable GPU: Runtime → Change runtime type → T4 GPU → Save
Step 3 — Upload SoliData.zip using the folder icon on the left sidebar. Run Section 2 of the notebook to extract it automatically.
Step 4 — Run all cells from top to bottom. The notebook will:

Install dependencies
Extract and load the dataset
Visualize range-Doppler maps
Define all model architectures from scratch
Train the conditional WGAN-GP on fine-grained classes (Phase 1)
Generate 200 synthetic samples per fine-grained class
Train the DANN classifier with GAN augmentation (Phase 2)
Evaluate per-class accuracy on both folds
Run ablation study comparing all four configurations
Download model checkpoints and plots


Pre-trained Model Weights
FileDescriptionLinkbest_model_fold0.ptBest DANN model Fold 1Add Google Drive link herebest_model_fold1.ptBest DANN model Fold 2Add Google Drive link heregenerator_final.ptTrained GAN generatorAdd Google Drive link here

Results
Main Results — Two-Fold Cross-Validation over Subjects
MetricFold 1Fold 2MeanOverall Accuracy90.61%95.12%92.86%Fine-Grained Accuracy78.17%93.43%85.80%
Per-Class Accuracy
ClassFold 1Fold 2Meanpinch_index (FG)86.00%89.14%87.57%pinch_pinky (FG)97.00%98.29%97.64%finger_slide (FG)76.00%94.29%85.14%finger_rub (FG)53.67%92.00%72.83%slow_swipe96.33%96.57%96.45%fast_swipe99.33%96.00%97.67%push99.33%98.29%98.81%pull98.00%98.29%98.14%palm_tilt99.33%98.86%99.10%circle95.33%96.57%95.95%palm_hold96.33%88.00%92.17%
FG = Fine-Grained class
Ablation Study
MethodOverall MeanFG Accuracy MeanBaseline (3D-CNN only)90.79%85.87%+ GAN augmentation87.89%84.16%+ DANN only84.94%73.46%Full (GAN + DANN)91.69%85.57%

Model Architecture

Input shape: (batch, 1, 40, 32, 32) — one radar clip
Encoder: Three 3D convolutional blocks → 256-dimensional feature vector
Gesture Classifier: Two linear layers → 11 output classes
Domain Discriminator: Two linear layers with Gradient Reversal Layer → 10 subjects
Generator (GAN): Noise + class embedding → 3D deconvolutional decoder → fake radar clip
GAN Discriminator: 3D strided convolutions → real/fake scalar


Hyperparameters
ParameterValueGAN training epochs50DANN training epochs30Synthetic samples per fine-grained class200Feature dimension256Domain loss weight (lambda)0.1GAN noise dimension100GAN learning rate2e-4DANN learning rate1e-3Gradient penalty coefficient10.0Batch size32Sequence length (frames)40

References

Wang et al. Interacting with Soli: Exploring Fine-Grained Dynamic Gesture Recognition in the Radio-Frequency Spectrum. ACM UIST 2016.
Ganin et al. Domain-Adversarial Training of Neural Networks. JMLR 2016.
Gulrajani et al. Improved Training of Wasserstein GANs. NeurIPS 2017.
Mirza and Osindero. Conditional Generative Adversarial Nets. arXiv 2014.
