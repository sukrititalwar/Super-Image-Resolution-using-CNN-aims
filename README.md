# Super Resolution using DCSCN

This repository contains code for 4x image super-resolution using a deep CNN architecture inspired by DCSCN.

## Folder Structure
- `dataset.py`: Dataset loader for LR-HR image pairs.
- `model.py`: DCSCN model architecture.
- `train.py`: Training script.
- `evaluate.py`: Evaluation script using PSNR and SSIM.
- `results/`: Directory to save model checkpoints and output.

## Dataset
Organize the dataset as follows:
```
data/
├── train/
│   ├── LR/
│   └── HR/
└── test/
    ├── LR/
    └── HR/
```

## Training
Run:
```
python train.py
```

## Evaluation
Run:
```
python evaluate.py
```

## Notes
- Upscaling factor: 4x
- Metrics used: PSNR, SSIM
