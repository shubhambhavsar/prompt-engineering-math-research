from setuptools import setup, find_packages

setup(
    name="gsm8k_research",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "datasets>=2.14.0",
        "pandas>=2.0.0",
        "requests>=2.31.0",
    ],
)