"""
AI Image Detector 패키지 설정 파일
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-image-detector",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI 생성 이미지와 실제 이미지를 분류하는 딥러닝 기반 판별기",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Ai-image-detector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)

