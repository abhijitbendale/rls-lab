Notes:
============


This package provides linear and non-linear rls code computation
in python similar to Ryan Rifkins matlab code. This code follows exactly the same
logic and steps to compute RLS. It is just python version of Ryan Rifkin's matlab
code.

With this package Ryan Rifkin's original code is added for reference. However,
for all the copyrights, licenses and IP related to original matlab code of 
Ryan Rifkin please refer his original code and copyright notice.

This python version is provided as is with no guarentee

data/smd.mat is a random matrix generated which was used for benchmarking purposes
The linear and non-linear RLS gives the same output as Ryan Rifkin's matlab code

Few tests and regression tests are provided with this package.

Example 1
-----------
python rls_pipeline.py data/smp.mat out --rls_type linear

python rls_pipeline.py data/smp.mat out --rls_type nonlinear

Tests
------------
cd tests/
nosetests test_rlspackage.py


