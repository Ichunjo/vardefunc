from setuptools import setup



requirements = open('requirements.txt').read()


name = 'vardefunc'
version = '1.0'

setup(
    name=name,
    version=version,
    author='Vardë',
    author_email="ichunjo.le.terrible@gmail.com",
    description="Vardë's Vapoursynth functions",
    long_description='README.md',
    url="https://github.com/Ichunjo/vardefunc",
    packages=['vardefunc'],
    install_requires=requirements,
    python_requires='>=3.8',
    zip_safe=False
)
