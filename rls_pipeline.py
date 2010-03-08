#!/usr/bin/python
#----------------------------------------------------------------------------------------
#This code follows GPL liecense
#
#Author: Abhijit Bendale
#        bendale@mit.edu
#        DiCarlo Lab,
#        Massachusetts Institute of Technology
#
#Date: July 7,2009
#
#Usage: python rls_io_interface.py <training_file> <linear/non-linear>
#Creates a file containing the list of best possible lambdas for given training set 
#using RLS.
#For more details about RLS, refer
#R.M.Rifkin, R.A.Lippert "Notes of Regularized Least Squares" CSAIL Tech Report, 
#MIT-CSAIL-TR-2007-025
#---------------------------------------------------------------------------------------

import sys
import optparse

import os.path as path
import scipy as sp
from scipy.io import (loadmat, savemat)
import scipy.linalg

#from mlabwrap import mlab
from utils.linearRLS import *
from utils.non_linear_rls import *
try:
    from utils.OptParserExtended import OptionExtended
except ImportError:
    print "OptParserExtended missing...!!!"


DEFAULT_RLS = 'linear'
DEFAULT_SAVE = False


# ------------------------------------------------------------------------------
def compute_rls(data_file,
                output_filename,
                rls_type = DEFAULT_RLS,
                save_out = DEFAULT_SAVE):
    """
    data file contains sampels and labels saved as a mat file
    with respective keywords. Make your own data wrapper if needed
    """

    if path.splitext(output_filename)[-1] != ".mat":
        output_filename += ".mat"        
        
    if path.splitext(data_file)[-1] != ".mat":
        raise ValueError, "mat file needed"

    data = loadmat(data_file)
    X = data['samples']
    Y = data['labels']
    lambdas = sp.logspace(-6,6,30)
    
    if rls_type.lower() == 'linear': 
        w,loos = lrlsloo(X, Y, lambdas)
    elif rls_type.lower() == 'nonlinear':
        w,loos = rlsloo(X, Y, lambdas)
    else:
        print "ERROR: specify linear or nonlinear"


    if save_out:
        out_data = {'weights': w,
                    'loos': loos}
        savemat(out_fname, out_data)

# ------------------------------------------------------------------------------
def main():
 
    usage = "usage: %prog [options]"
    usage += "<data file (labels + samples)> <outfname>"

    parser = optparse.OptionParser(usage=usage, option_class=OptionExtended)    
    parser.add_option("--rls_type",
                      type="string",
                      metavar="STRING",
                      action="store",
                      default=DEFAULT_RLS,
                      help="output directory for DET results[default=%default]")

    parser.add_option("--save_out",
                      default=DEFAULT_SAVE,
                      action="store_true",
                      help="overwrite existing file [default=%default]")

    opts, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
    else:
        data_file = args[0]
        output_filename = args[1]

        compute_rls(data_file,
                    output_filename,
                    rls_type = opts.rls_type,
                    save_out = opts.save_out)
        

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
