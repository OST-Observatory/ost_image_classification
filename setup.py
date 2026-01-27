from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
install_requires = []
if os.path.exists(requirements_path):
    with open(requirements_path, "r", encoding="utf-8") as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ost-image-classification",
    version="1.0.0",
    description="Multimodal image classifier for astronomical data (FITS/TIFF)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="OST-Observatory",
    url="https://github.com/OST-Observatory/ost_image_classification",  # Update with your repo URL
    py_modules=[
        "data_loader",
        "model",
        "main",
        "evaluate_model",
        "inference_runner",
    ],
    install_requires=install_requires,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "ost-classify=inference_runner:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    include_package_data=True,
    package_data={
        "": [
            "thresholds_*.json",
            "*.md",
        ],
    },
)
