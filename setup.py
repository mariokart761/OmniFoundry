"""
OmniFoundry 安裝腳本
"""

from setuptools import setup, find_packages
from pathlib import Path

# 讀取 README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# 讀取 requirements
requirements = (this_directory / "requirements.txt").read_text(encoding="utf-8").splitlines()

setup(
    name="omnifoundry",
    version="0.1.0",
    author="OmniFoundry Team",
    author_email="",
    description="開源模型推論環境集成程式 - 自動偵測硬體、智能選擇推論引擎的 AI 模型管理與部署工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/omnifoundry",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "omnifoundry=omnifoundry.cli.main:cli",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "llama-cpp": [
            "llama-cpp-python>=0.2.0",
        ],
        "optimum": [
            "optimum[onnxruntime]>=1.12.0",
        ],
        "all": [
            "llama-cpp-python>=0.2.0",
            "optimum[onnxruntime]>=1.12.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

