#!/usr/bin/env python
"""Setup script for the GTgen package"""

import codecs
import setuptools
import GTgen

setuptools.setup(
    # general description
    name='GTgen',
    description='Generate random graphs and timeseries, with anomalies',
    #version=tde.__version__,
    long_description=open('README.md').read(),
    license='LICENSE.txt',

    # python package dependencies
    #setup_requires=['pandas',
    #                'numpy'],
    #install_requires=['editdistance',
    #                  'joblib',
    #                'intervaltree'],

    tests_require=['pytest'],

    # packages for code and data
    packages=setuptools.find_packages(),
    #package_data={'tde': ['share/*']},

    # metadata for upload to PyPI
    author='Julien Karadayi, LIP6',
    author_email='julien.karadayi@lip6.fr',
    zip_safe=True,
)
