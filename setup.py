#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import setup
import numpy as np

setup(
        name = 'graph_library',
        version = '1.0',
        include_dirs = [np.get_include()], #Add Include path of numpy
        packages=['.'],
        install_requires=['numpy', 'scipy', 'networkx'],
      )
