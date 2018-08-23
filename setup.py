#!/usr/bin/env python

from setuptools import setup

setup(name='GENDIS',
      version='1.0.9',
      description=' Contains an implementation (sklearn API) of the algorithm proposed in "GENDIS: GEnetic DIscovery of Shapelets" and code to reproduce all experiments.',
      author='Gilles Vandewiele',
      author_email='gilles.vandewiele@ugent.be',
      url='https://github.com/IBCNServices/GENDIS',
      packages=['gendis'],
      install_requires=[
          'deap==1.2.2',
          'matplotlib==2.1.2',
      	  'numpy==1.14.5',
      	  'pandas==0.22.0',
      	  'pathos==0.2.2',
      	  'scipy==1.1.0',
      	  'sklearn==0.0',
      	  'tqdm==4.23.2',
          'Cython == 0.28.2',
      	  'tslearn==0.1.18.3',
          'nose2==0.8.0'
      ],
      test_suite='nose2.collector.collector',
     )
