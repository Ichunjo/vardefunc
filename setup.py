from setuptools import setup


with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = fh.read()

NAME = "vardefunc"
VERSION = "1.1.2"

setup(
    name=NAME,
    version=VERSION,
    author="Vardë",
    author_email="ichunjo.le.terrible@gmail.com",
    description="Vardë's Vapoursynth functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ichunjo/vardefunc",
    packages=["vardefunc"],
    install_requires=install_requires,
    python_requires=">=3.8",
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
