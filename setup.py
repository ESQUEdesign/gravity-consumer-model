from setuptools import setup, find_packages

setup(
    name="gravity-consumer-model",
    version="0.1.0",
    description="Layered consumer behavior prediction engine built on the Huff Retail Gravity Model",
    author="Rosai",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "geopandas>=0.13",
        "scikit-learn>=1.3",
        "statsmodels>=0.14",
        "pydantic>=2.0",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "hmmlearn>=0.3",
        "folium>=0.14",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "osmnx>=1.5",
        "libpysal>=4.7",
        "spreg>=1.4",
        "pyyaml>=6.0",
        "shapely>=2.0",
    ],
    extras_require={
        "gnn": ["torch>=2.0", "torch-geometric>=2.3"],
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
)
