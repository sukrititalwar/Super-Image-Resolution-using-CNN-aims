![image](https://github.com/user-attachments/assets/405eb66e-1578-4c22-b604-1d94ffc38a37)# Super Resolution using DCSCN

This project explores image super-resolution using the Deep CNN with Skip Connection and Network in Network (DCSCN) model. The objective is to enhance low-resolution images to higher resolutions with better visual quality. Through this work, we learned how deep learning-based architectures can reconstruct high-frequency details and how model training, preprocessing, and evaluation play crucial roles in image quality enhancement

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
