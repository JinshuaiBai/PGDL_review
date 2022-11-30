"""

    This function is to visualize the displacment and stress contours.
        
"""
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

def Vis(pinn, net1, T, L, it):
    """
    =================================================================================================================================
    
    Options:
        Name        Type                    Size        Info.
        
        'E'         [float]                 1           : Young's module;
        'pinn'      [keras model]           \           : The trained PINN;
        'net1'      [keras model]           \           : The trained FNN for displacment u;
        'net2'      [keras model]           \           : The trained FNN for displacment v;
        'xy'        [Array of float32]      ns*2        : The sample points coordinates for visualization.
        
    Variables:
        Name        Type                    Size        Info.
        
        [u]         [Array of float32]      ns*1        : Displacement u;
        [v]         [Array of float32]      ns*1        : Displacement v;
        [s11]       [Array of float32]      ns*1        : Nornal stress sigma_x;
        [s22]       [Array of float32]      ns*1        : Nornal stress sigma_y;
        [s12]       [Array of float32]      ns*1        : Shear stress tau_xy.

    =================================================================================================================================    
    """
    
    xy = np.zeros((101*101, 2)).astype(np.float32)
    for i in range(0,101):
        for j in range(0,101):
            xy[i*101+j,0] = i * 2/100-1
            xy[i*101+j,1] = j * 2/100-1
    u = net1.predict(xy)
    
    # plot figure for displacement u
    fig1 = plt.figure(1)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = u, cmap = 'jet')
    plt.axis('equal')
    plt.colorbar()
    plt.title('u')
    

    scipy.io.savemat('out.mat', {'xy': xy, 'u': u}) 