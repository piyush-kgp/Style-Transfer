
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [line.strip() for line in open('requirements.txt', 'r').readlines()]

setuptools.setup(
    name="style_transfer",
    version="0.0.4",
    author="Piyush Singh",
    author_email="piyushsinghkgpian@gmail.com",
    description="A library for style transfer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
