import os
from setuptools import setup, find_packages

# Safe README loading
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read() if os.path.exists("README.md") else ""

setup(
    name="TwoPlayerGame",
    version="1.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch>=2.0",
        "numpy>=1.21",
        "matplotlib>=3.4",
    ],
    author="Melih Iseri",
    description="A Python package simulating a two-player repeated game with action learning and cost estimation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TwoPlayerGame",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
