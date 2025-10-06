## Conditional Flow Matching for Image Quality Transfer - An application to Low-Field MRI
This repository contains a PyTorch implementation of a Conditional Flow Matching (CFM) model for 2D MRI image enhancement. The model utilizes a U-Net based architecture to learn the transformation from low-resolution to high-resolution medical images.

## Project Structure
The project is organized to separate configuration, source code, and results, promoting modularity and ease of use.

cfm_project/
│
├── dataset/
│   ├── Training/                 # Training data
│   │   ├── LR/                   # Low-resolution images
│   │   └── HR/                   # High-resolution images
│   └── Testing/                  # Testing data
│       ├── InD/                  # In-distribution test set
│       └── OOD/                  # Out-of-distribution test set
│
├── output/                       # Evaluation results
│
├── src/
│   ├── config.yaml               # Main configuration file: hyperparameters, paths, seed, etc.
│   ├── dataset.py                # Dataloader and MRI-specific augmentation
│   ├── model.py                  # CFM U-Net model definition
│   ├── solver.py                 # ODE solver for CFM inference
│   ├── test.py                   # Script for evaluating the model
│   ├── train.py                  # Script for training the model
│   └── utils.py                  # Utility functions: metrics, plotting, seed setup, etc.
│
├── weights/                      # Directory for trained model checkpoints
│
├── requirements.txt              # Python package dependencies
│
└── README.md                     # Project description and usage instructions

### Dataset
I use a dataset of three-dimensional high-resolution T1-weighted (T1w) MRI images provided
by the Human Connectome Project (HCP) (https://www.sciencedirect.com/science/article/pii/S105381191300551X) dataset to train and test the model.

### Train
```shell
python src/train.py \
    --config src/config.yaml \
    --epochs 10 \
    --learning_rate \
    --weight_decay \
    --batch_size
```

### Test
```shell
python src/test.py \
    --config src/config.yaml \
    --mode ood_testing_phase \
    --batch_size 4 \
    --t_steps 10
```

### Updates
- **14 August, 2025**: Initialize.
