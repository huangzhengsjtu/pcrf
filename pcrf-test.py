# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:36:01 2014

@author: Huang,Zheng

A command line wrapper to use LinearCRF class to Test.

"""

import argparse
import LinearCRF2 

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="data file for Testing input")
    parser.add_argument("modelfile", 
                        help="the learnt model file to load. ")
    parser.add_argument("resultfile", 
                        help="the output file name.")
    #parser.add_argument("-s", "--sigma", type=float, 
    #                    default=1,
    #                    help="sigma")
    args = parser.parse_args()
    
    LinearCRF2.crfpredict(args.datafile, args.modelfile, args.resultfile)
   
 