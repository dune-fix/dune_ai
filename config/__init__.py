from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="dune-ai",
    version="0.1.0",
    author="DUNE AI Team",
    author_email="support@duneai.io",
    description="AI-powered Solana meme coin analytics and trend detection system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duneai/dune-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "dune-ai=dune_ai.main:main",
        ],
    },
    include_package_data=True,
)