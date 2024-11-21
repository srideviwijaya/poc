FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Install PyTorch
RUN pip install torchvision torchaudio torchtext datasets argparse

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
