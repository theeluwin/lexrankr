from typing import Tuple

from setuptools import (
    setup,
    find_packages,
)


requirements: Tuple[str, ...] = (
    'setuptools',
    'networkx',
    'scipy',
    'numpy',
    'gensim',
    'sklearn',
)


setup(
    name='lexrankr',
    version='1.0',
    license='MIT',
    author='Jamie Seol',
    author_email='theeluwin@gmail.com',
    url='https://github.com/theeluwin/lexrankr',
    description="LexRank based multi-document summarization.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[],
)
