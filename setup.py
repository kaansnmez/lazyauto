#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages, Extension

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
    long_description_content_type="text/markdown",
    #long_description=readme + '\n\n' + history,
    long_description='You can find example in my github: '+'https://github.com/kaansnmez/lazyauto',
    include_package_data=True,
    keywords=['lazyauto','pipeline','optuna'],
    name='lazyauto',
    packages=find_packages(include=['lazyauto', 'lazyauto.*']),
    test_suite='tests',
    download_url='https://github.com/user/lazyauto/archive/v_01.tar.gz',
    tests_require=test_requirements,
    url='https://github.com/kaansnmez/lazyauto',
    version='0.2.5',
    zip_safe=False,
)
