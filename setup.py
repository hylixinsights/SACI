from setuptools import setup, find_packages

setup(
    name="saci",
    version="0.3.0",
    description="SACI (Single-cell Adaptive Clustering and Identification): distribution-aware hierarchical gene selection and co-expression rescue for scRNA-seq",
    author="",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
        "tqdm>=4.60",
        "anndata>=0.8",
        "diptest>=0.6",
        "igraph>=0.10",
    ],
    extras_require={
        "scanpy": ["scanpy>=1.9"],
        "leiden": ["leidenalg>=0.9"],
    },
)

