#!/usr/bin/env python

from setuptools import setup
import numpy as np
from Cython.Build import cythonize

setup(name='GENDIS',
      version='1.0.13',
      description=' Contains an implementation (sklearn API) of the algorithm proposed in "GENDIS: GEnetic DIscovery of Shapelets" and code to reproduce all experiments.',
      author='Gilles Vandewiele',
      author_email='gilles.vandewiele@ugent.be',
      url='https://github.com/IBCNServices/GENDIS',
      packages=['gendis'],
      install_requires=[
          'deap==1.2.2',
          'matplotlib==2.1.2',
      	  'numpy==1.22.0',
      	  'pandas==0.22.0',
      	  'pathos==0.2.2',
      	  'scipy==1.1.0',
      	  'sklearn==0.0',
      	  'tqdm==4.23.2',
      	  'tslearn==0.1.27',
          'nose2==0.8.0',
          'terminaltables==3.1.0'
      ],
      test_suite='nose2.collector.collector',
      ext_modules = cythonize(['gendis/pairwise_dist.pyx']),
      include_dirs=[np.get_include()]
     )
