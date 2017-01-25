"""Tinyik is a simple and naive inverse kinematics solver."""

from setuptools import setup, find_packages
from codecs import open


with open('README.rst', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='tinyik',
    version='1.2.0',
    description='A tiny inverse kinematics solver',
    long_description=long_description,
    url='http://github.com/lanius/tinyik',
    author='lanius',
    author_email='lanius@nirvake.org',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    packages=find_packages(exclude=['tests']),
    install_requires=['autograd', 'scipy'],
)
