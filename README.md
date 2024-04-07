# Diffusion Model using TensorFlow

This repository provides TensorFlow 2 implementation for unconditional image generation, utilizing the following methods:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)

## Set-up

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

## File Structure

- `dataset/`: Contains modules for loading and preprocessing training data.
- `difussion/`: Modules for the diffusion process (e.g., forward process, reverse step using DDIM/DDPM, variance scheduling).
- `model/`: Helper modules for the UNet architecture.
- `train.py`: Script for training the reverse process.
- `train_distill.py`: Script for distilling the original model using progressive distillation.
- `infer.py`: Script for running the inference using pre-trained model.
- `utils.py`: Module containing utility functions.
- `configs.py`: Helper module for loading configurations.

## Training and Inference

The configurations for the diffusion process, dataset, UNet model, and training/inference related settings are present in `configs.yaml`. Once updated, train the original/teacher model using:

```bash
python train.py
```

For distilling the student model, run the following after changing the
`train_cfg.teacher_checkpoint` in `configs.yaml`:

```bash
python train_distill.py
```

For inference, update `train_cfg.checkpoint` in `configs.yaml` with the original or distilled model's checkpoint configs, then run:

```bash
python infer.py
```
