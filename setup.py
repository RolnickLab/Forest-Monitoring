from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()
keywords = ["segmentation", "deep learning", "forest monitoring", "dataset", "pytorch"]

setup(
    name="treemonitoring",
    version="0.1.0",
    packages=find_packages(),
    url="https://github.com/RolnickLab/Forest-Monitoring",
    license="CC BY 4.0",
    author="Venkatesh Ramesh, Arthur Ouaknine and David Rolnick",
    author_email="",
    keywords=keywords,
    description="The code repository for our work Tree semantic segmentation from aerial image time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)