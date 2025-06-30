from setuptools import setup, find_packages

setup(
    name="events_analyzer",
    version="1.0.0",
    description="Анализ и кластеризация мероприятий",
    author="Набиев Рашидхон",
    python_requires=">=3.12",
    packages=find_packages(where="src"),   # ← ищем пакеты внутри src/
    package_dir={"": "src"},               # ← всё находится в каталоге src
    install_requires=[
        "pandas",
        "sentence-transformers",
        "scikit-learn",
        "hdbscan",
        "matplotlib",
    ],
)