from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()
keywords = ["segmentation", "deep learning", "tree-monitoring", "dataset", "pytorch", "mila"]

setup(
    name="treemonitoring",
    version="0.1.0",
    packages=find_packages(),
    url="https://github.com/RolnickLab/Tree-Monitoring",
    license="CC BY 4.0",
    author="Venkatesh Ramesh, Arthur Ouaknine",
    author_email="",
    keywords=keywords,
    description="Forest-monitoring to study the behaviour of trees from spring to leaf-off season.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    #    install_requires=[
    #        "torch>=1.13.0",
    #    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
