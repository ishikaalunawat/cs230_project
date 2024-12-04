# DeepSee: Marine Species Detection in Underwater Environments
Code for CS 230 Project Autumn 2024.

## Files
### Outputs:
1. `outputs_/`: Outputs (train logging, visualizations, etc) from MAD denoiser
### Denoiser:
1. `trainer_denoiser.py`: The trainer class for denoiser models MAD and Res-U-Net
2. `datasets_denoiser.py`: Pytorch dataset for denoiser models MAD and Res-U-Net
3. `main_denoiser.py`: Main script to train (and evaluate) denoiser models MAD and Res-U-Net
4. `losses_denoiser.py`: Loss functions defined in report for denoiser models MAD and Res-U-Net
5. `create_noisy_data.ipynb`: To create noisy data pairs for our dataset
6. `test_denoiser.py`: To test, and save vizuals for denoiser models MAD and Res-U-Net

### Classification
1. `train_yolo.ipynb`: Train YOLO
2. `train_resnet18.ipynb`: Train ResNet18
3. `train_cnn.ipynb`: Train simple CNN
4. `trainer.py`: Trainer module for classification and some test denoisers

