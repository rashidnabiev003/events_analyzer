from setuptools import setup, find_packages

setup(
    name="events_analyzer",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "pandas",
        "sentence-transformers",
        "scikit-learn",
        "hdbscan",
        "matplotlib",
    ],
)