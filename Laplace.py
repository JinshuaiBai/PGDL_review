"""

    This code is for example used in "Physics-guided deep learning for data scarcity." 
    DOI: https://doi.org/10.48550/arXiv.2211.15664

    This code is developed by @Jinshuai Bai and @Yuantong Gu. For more details, please contact: 
    jinshuai.bai@hdr.qut.edu.au
    yuantong.gu@qut.edu.au.
        
"""
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import scipy.io
from Input_info import Input
from FNN import Net
from PINN import PINN
from OPT_PINN import OPT_PINN
from OPT_DL import OPT_DL
from Visualization import Vis

#%%
if __name__ == '__main__':
    """
    
        Input information for calculations.
        
    """
    
    ns, ns_u, ns_l, x_train, y_train, dx= Input()
    
    """
    
        Build two Feedforward Neural Networks (FNN) to respectively predict the displacment u and v.
        
    """
    
    net1 = Net(n_input = 2, n_output = 1, layers=[20,20,20,20])
    
    """
    
        Build a Physics-Informed Neural Networks (PINN) by the pre-built neural network.
        
    """
    
    pinn = PINN(net1)
    
    """
    
        Initialize the L-BFGS-B optimizer.
        
    """

    opt_pinn = OPT_PINN(pinn, x_train, y_train, dx)
    
    """
    
        Train the PINN through the L-BFGS-B optimizer. Print the training time, final loss, and overall
        iterations for convergence. 
        
    """
    
    time_start = time.time()
    result = opt_pinn.fit()
    time_end = time.time()
    
    T = time_end-time_start
    L = result[1]
    it = result[2]['funcalls']
    print('-------------------------------------------------\n')
    print('Time cost is', T, 's')
    print('Final loss is', L, '')
    print('Training converges by', it, 'iterations\n')
    print('-------------------------------------------------\n')
    
#%%

    net2 = Net(n_input = 2, n_output = 1, layers=[20,20,20,20])
    opt_dl = OPT_DL(net2, x_train[1], y_train, dx)
    
    time_start = time.time()
    result = opt_dl.fit()
    time_end = time.time()
    
    T = time_end-time_start
    L = result[1]
    it = result[2]['funcalls']
    print('-------------------------------------------------\n')
    print('Time cost is', T, 's')
    print('Final loss is', L, '')
    print('Training converges by', it, 'iterations\n')
    print('-------------------------------------------------\n')

#%%
    xy = np.zeros((101*101, 2)).astype(np.float32)
    for i in range(0,101):
        for j in range(0,101):
            xy[i*101+j,0] = i * 2/100-1
            xy[i*101+j,1] = j * 2/100-1
    u_pinn = net1.predict(xy)
    u_dl = net2.predict(xy)
    
    # plot figure for u_pinn
    fig1 = plt.figure(1)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = u_pinn, cmap = 'jet', vmin=-0.4, vmax=0.4)
    plt.axis('equal')
    plt.colorbar()
    plt.title('u')
    
    # plot figure for u_dl
    fig1 = plt.figure(2)
    plt.scatter(xy[:,0], xy[:,1], s = 5, c = u_dl, cmap = 'jet', vmin=-0.4, vmax=0.4)
    plt.axis('equal')
    plt.colorbar()
    plt.title('u')
