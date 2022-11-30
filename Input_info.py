"""

    This function is for initialize all the variables for the calculation.
        
"""
import numpy as np
import math
import matplotlib.pyplot as plt

def Input():
    """
    =================================================================================================================================
    
    Variables:
        Name        Type                    Size        Info.
        
        [ns]        [int]                   1           : Total number of sample points;
        [ns_u]      [int]                   1           : Number of sample points on top boundary of the beam;
        [ns_l]      [int]                   1           : Number of sample points on left boundary of the beam;
        [dx]        [float]                 1           : Sample points interval;
        [h]         [float]                 1           : Parameter relates to sample points interval;
        [xy]        [Array of float32]      ns*2        : Coordinates of all the sample points;
        [xy_u]      [Array of float32]      ns_u*2      : Coordinates of the sample points on the top boundary of the beam;
        [xy_b]      [Array of float32]      ns_u*2      : Coordinates of the sample points on the bottom boundary of the beam;
        [xy_l]      [Array of float32]      ns_l*2      : Coordinates of the sample points on the left boundary of the beam;
        [xy_r]      [Array of float32]      ns_l*2      : Coordinates of the sample points on the right boundary of the beam;
        [x_train]   [List]                  5           : PINN input list, contains all the coordinates information;
        [s_u_x]     [Array of float32]      ns_u*1      : x direction force boundary condition on the top boundary of the beam;
        [s_u_y]     [Array of float32]      ns_u*1      : y direction force boundary condition on the top boundary of the beam;
        [s_b_x]     [Array of float32]      ns_u*1      : x direction force boundary condition on the bottom boundary of the beam;
        [s_b_y]     [Array of float32]      ns_u*1      : y direction force boundary condition on the bottom boundary of the beam;
        [s_l_x]     [Array of float32]      ns_l*1      : x direction force boundary condition on the left boundary of the beam;
        [s_l_y]     [Array of float32]      ns_l*1      : y direction force boundary condition on the left boundary of the beam;
        [s_r_x]     [Array of float32]      ns_l*1      : x direction force boundary condition on the right boundary of the beam;
        [s_r_y]     [Array of float32]      ns_l*1      : y direction force boundary condition on the right boundary of the beam;
        [y_train]   [List]                  8           : PINN boundary condition list, contains all the force boundary conditions;
        [E]         [float]                 1           : Young's module;
        [mu]        [float]                 1           : Poisson ratio.
        
    =================================================================================================================================    
    """
    print('-------------------------------------------------\n')
    print('                 Input Info.\n')
    print('-------------------------------------------------\n')
    
    # number of sample points
    
    ns_u = 11
    ns_l = ns_u
    ns = ns_u*ns_u
    
    # sample points' interval
    dx = 2/(ns_u-1)
    
    print(ns, 'sample points')
    print(ns_u,'sample points on the top boundary;',ns_u,'sample points on the bottom boundary;')
    print(ns_l,'sample points on the left boundary;',ns_l,'sample points on the right boundary.')
    
    # initialize sample points' coordinates    
    xy = np.zeros((ns, 2)).astype(np.float32)
    for i in range(0,ns_u):
        for j in range(0,ns_l):
            xy[i*ns_l+j,0] = i * dx - 1. 
            xy[i*ns_l+j,1] = j * dx - 1.
    xy_t = np.hstack([np.linspace(-1,1, ns_u).reshape(ns_u, 1).astype(np.float32), \
                      np.ones((ns_u,1)).astype(np.float32)])
    xy_b = np.hstack([np.linspace(-1,1, ns_u).reshape(ns_u, 1).astype(np.float32), \
                      -1*np.ones((ns_u,1)).astype(np.float32)])
    xy_l = np.hstack([-1*np.ones((ns_l,1)).astype(np.float32), \
                  np.linspace(-1,1, ns_l).reshape(ns_l, 1).astype(np.float32)])
    xy_r = np.hstack([np.ones((ns_l,1)).astype(np.float32), \
                  np.linspace(-1,1, ns_l).reshape(ns_l, 1).astype(np.float32)])
    xy_bound = np.vstack([xy_t, xy_b, xy_l, xy_r])
    # create PINN input list
    x_train = [ xy, xy_bound ]
    
    # boundary conditions
    ge = -np.cos(math.pi*xy[...,0,np.newaxis]/2)*np.sin(math.pi*xy[...,1,np.newaxis])*5
    u_bound = np.zeros((ns_u*4,1)).astype(np.float32)

    # create PINN boundary condition list
    y_train = [ge,u_bound]
    
    return ns, ns_u, ns_l, x_train, y_train, dx