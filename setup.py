from setuptools import setup



requirements = open("requirements.txt").read()


NAME = "vardefunc"
VERSION = "1.0"

setup(
    name=NAME,
    version=VERSION,
    author="Vardë",
    author_email="ichunjo.le.terrible@gmail.com",
    description="Vardë's Vapoursynth functions",
    long_description="README.md",
    url="https://github.com/Ichunjo/vardefunc",
    download_url="https://github.com/Ichunjo/vardefunc/archive/refs/tags/v1.0-beta.zip",
    packages=["vardefunc"],
    install_requires=requirements,
    python_requires=">=3.8",
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
