#!/usr/bin/env python                                                                                                                                                                                           

from distutils.core import setup, Extension
import glob
import os

# Get matfiles and images for testing
matfiles=glob.glob(os.path.join('tests/data/*.mat'))
data=glob.glob(os.path.join('data/*'))


setup(
    name='pyphasesym',
    version='1.0',
    description='Python implementation of phasesym program',
    author='Abhijit Bendale',
    author_email='bendale@mit.edu',
    py_modules = ['rls_pipeline','tests.test_rlspackage',
                  'utils.linearRLS', 'utils.non_linear_rls',
		 'OptParserExtended'],
    data_files = [('documentation',['documentation/notes.rst']),
		  ('data', ['data/smp.mat']),
		  ('tests/data', ['tests/data/smp.mat','tests/data/linear_rls.mat', 'tests/data/non_linear_rls.mat'])],

    )
    
