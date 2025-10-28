from setuptools import setup, find_packages

setup(
    name="stable-diffusion-pytorch",
    version="1.0.0",
    description="PyTorch implementation of Stable Diffusion",
    author="",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "einops>=0.6.0",
        "pillow>=9.0.0",
        "accelerate>=0.20.0",
    ],
    python_requires=">=3.8",
)

