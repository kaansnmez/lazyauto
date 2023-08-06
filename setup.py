#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pandas','scikit-learn','matplotlib','optuna','seaborn','catboost','numpy','plotly','xgboost','lightgbm' ]

test_requirements = ['pytest>=3', ]

setup(
    author="Ahmet Kaan Sonmez",
    author_email='kaan.sonmez97@gmail.com',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A python package for analysis and model development.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='lazyauto',
    name='lazyauto',
    packages=find_packages(include=['lazyauto', 'lazyauto.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/kaansnmez/lazyauto',
    version='0.1.0',
    zip_safe=False,
)
