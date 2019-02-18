
"""
A Library for Style Transfer
"""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['requests==2.19.1', 'tensorflow==1.12.0', 'matplotlib==2.2.2']

setup(name='style_transfer',
      version='1.0',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=find_packages(),
      description='A library for style transfer'
)
