from setuptools import setup, find_packages

setup(
    name="neuralnetworkbasics",
    version="0.1.0",
    description="Build neural networks using pail python library",
    author="Sunil Tiwari",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
)
