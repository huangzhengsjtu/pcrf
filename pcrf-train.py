# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:36:01 2014

@author: Huang,Zheng

A command line wrapper to use LinearCRF to train.

"""

import argparse
import LinearCRF2 

if __name__ ==  '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile", help="data file for training input")
    parser.add_argument("templatefile", 
                        help="template file for generate feature functions.")
    parser.add_argument("modelfile", 
                        help="the learnt model file. (output)")
    parser.add_argument("-r", "--regularity", type=int, 
                        default=2, choices=[0, 1, 2],
                        help="regularity: 0:none;  1:first order;  2:square.")
    parser.add_argument("-s", "--sigma", type=float, 
                        default=1,
                        help="sigma")
    parser.add_argument("-m", "--multiproc", type=int, 
                        default=1, choices=[0,1],
                        help="multiprocessing: 1:use multiprocessing; 0:only single core.")
    parser.add_argument("-f", "--fd", type=int, 
                        default=1, 
                        help="feature reduction: the number of observed x under this value is ignored.")
    
    args = parser.parse_args()

    #print args.sigma

    LinearCRF2.train(args.datafile,args.templatefile,args.modelfile,
        regtype=args.regularity,sigma=args.sigma,mp=args.multiproc, fd=args.fd)
   

    
