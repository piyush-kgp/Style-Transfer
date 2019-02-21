
"""
A Library for Style Transfer
"""

# from setuptools import find_packages
# from setuptools import setup
#
# REQUIRED_PACKAGES = ['requests==2.19.1', 'tensorflow==1.12.0', 'matplotlib==2.2.2']
#
# setup(name='style_transfer',
#       version='1.0',
#       install_requires=REQUIRED_PACKAGES,
#       include_package_data=True,
#       packages=find_packages(),
#       description='A library for style transfer'
# )


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = ['requests==2.19.1', 'tensorflow==1.12.0', 'matplotlib==2.2.2']
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
