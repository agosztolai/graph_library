#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages

setup(
        name = 'graph_library',
        version = '1.0',
        packages=find_packages(),
        install_requires=['numpy', 
                          'scipy', 
                          'networkx',
                          'scikit-learn',
                          'pyyaml',
                          'matplotlib'],
        include_package_data = True
      )
