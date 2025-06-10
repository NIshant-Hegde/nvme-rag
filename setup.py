#!/usr/bin/env python3

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from file
def read_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file) as f:
            lines = f.readlines()
        
        # Filter out comments and empty lines
        requirements = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove version specifiers that might cause issues
                if line.startswith('-'):
                    continue
                requirements.append(line)
        return requirements
    return []

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="nvme-rag",
    version="1.0.0",
    description="Professional Retrieval-Augmented Generation system for NVMe specifications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NVMe RAG Development Team",
    author_email="nvme-rag@example.com",
    url="https://github.com/example/nvme-rag",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "nvme-rag=nvme_rag.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Documentation",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="rag retrieval-augmented-generation nvme nlp ai machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/example/nvme-rag/issues",
        "Source": "https://github.com/example/nvme-rag",
        "Documentation": "https://github.com/example/nvme-rag/blob/main/README.md",
    },
)
