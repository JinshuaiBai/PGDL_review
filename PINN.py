"""

    This function is to initialize a PINN.
        
"""
import tensorflow as tf
from Dif_op import DIF

def PINN(net):
    """
    =================================================================================================================================
    
    Options:
        Name        Type                    Size        Info.
        
        'net1'      [keras model]           \           : The trained FNN for displacment u;
        'net2'      [keras model]           \           : The trained FNN for displacment v;
        'mu'        [float]                 1           : Poisson ratio.
        
    Variables:
        Name        Type                    Size        Info.
        
        [xy]        [Array of float32]      ns*2        : Coordinates of all the sample points;
        [xy_u]      [Array of float32]      ns_u*2      : Coordinates of the sample points on the top boundary of the beam;
        [xy_b]      [Array of float32]      ns_u*2      : Coordinates of the sample points on the bottom boundary of the beam;
        [xy_l]      [Array of float32]      ns_l*2      : Coordinates of the sample points on the left boundary of the beam;
        [xy_r]      [Array of float32]      ns_l*2      : Coordinates of the sample points on the right boundary of the beam;
        [xy_f]      [Array of float32]      1*2         : Coordinates of the fixed sample point;
        [Gex]       [Array of float32]      ns*1        : Equilibrium equation for x direction of internal computational domain;
        [Gey]       [Array of float32]      ns*1        : Equilibrium equation for y direction of internal computational domain;
        [Gex_u]     [Array of float32]      ns_u*1      : Equilibrium equation for x direction on the top boundary of the beam;
        [Gey_u]     [Array of float32]      ns_u*1      : Equilibrium equation for y direction on the top boundary of the beam;
        [Gex_b]     [Array of float32]      ns_u*1      : Equilibrium equation for x direction on the bottom boundary of the beam;
        [Gey_b]     [Array of float32]      ns_u*1      : Equilibrium equation for y direction on the bottom boundary of the beam;
        [Gex_l]     [Array of float32]      ns_l*1      : Equilibrium equation for x direction on the left boundary of the beam;
        [Gey_l]     [Array of float32]      ns_l*1      : Equilibrium equation for y direction on the left boundary of the beam;
        [Gex_r]     [Array of float32]      ns_l*1      : Equilibrium equation for x direction on the right boundary of the beam;
        [Gey_r]     [Array of float32]      ns_l*1      : Equilibrium equation for y direction on the right boundary of the beam;
        [s_u_x]     [Array of float32]      ns_u*1      : x direction force boundary condition on the top boundary of the beam;
        [s_u_y]     [Array of float32]      ns_u*1      : y direction force boundary condition on the top boundary of the beam;
        [s_b_x]     [Array of float32]      ns_u*1      : x direction force boundary condition on the bottom boundary of the beam;
        [s_b_y]     [Array of float32]      ns_u*1      : y direction force boundary condition on the bottom boundary of the beam;
        [s_l_x]     [Array of float32]      ns_l*1      : x direction force boundary condition on the left boundary of the beam;
        [s_l_y]     [Array of float32]      ns_l*1      : y direction force boundary condition on the left boundary of the beam;
        [s_r_x]     [Array of float32]      ns_l*1      : x direction force boundary condition on the right boundary of the beam;
        [s_r_y]     [Array of float32]      ns_l*1      : y direction force boundary condition on the right boundary of the beam;
        [mu]        [float]                 1           : Poisson ratio.

    =================================================================================================================================    
    """

    ### Declare inputs
    xy = tf.keras.layers.Input(shape=(2,))
    xy_bound = tf.keras.layers.Input(shape=(2,))
    
    ### Initialize the differential operators
    dif = DIF(net)
    
    ### Obtain partial derivatives with respect to x and y
    U_xx, U_yy = dif(xy)
    
    ### Obtain the residuals from stress boundary conditions
    u_bound = net(xy_bound)
    
    ### Obtain the residuals from equilibrium equation
    Ge = U_xx + U_yy
    
    return tf.keras.models.Model(
        inputs = [xy, xy_bound], \
            outputs = [Ge,u_bound])