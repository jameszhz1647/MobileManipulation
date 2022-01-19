import numpy as np
from numpy import arctan2
import modern_robotics as mr
from modern_robotics import MatrixExp6, VecTose3, ScrewTrajectory, CartesianTrajectory
import csv

def NextState(config_cur, vel, dt, v_lim):
    """A simulator for the kinematics
    Inputs
    :param config_cur: A 12-vector representing the current configuration of the robot
    :param vel: A 9-vector of controls indicating the arm joint speeds
    :param dt: timestep 
    :param v_lim: A positive real value indicating the maximum angular speed of the arm joints and the wheels. (float or int)
                      Or A list of positive real value indicating the maximum angular speed of the each arm joints and the wheels. (List(float or int))
    Output
    :param config_new: A 12-vector representing the configuration of the robot later.
    """
    # car parameters
    r = 0.0475 # radius of wheel
    l = 0.47 / 2 # half length
    w = 0.3 / 2 # half wide
    h = 0.0963 # height
    
    #find pseudo-inverse H for chassis 
    H_pse = r / 4 * np.array([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                              [1, 1, 1, 1],
                              [-1, 1, -1, 1],
                              [0, 0, 0, 0]])
    # find chassis twist
    V_cha = H_pse.dot(vel[5:9])
    # find transforamtion in chassis frame
    T_cha2cha_new = MatrixExp6(VecTose3(V_cha*dt))
    T_cha = np.array([[np.cos(config_cur[0]), -np.sin(config_cur[0]), 0, config_cur[1]], 
                     [np.sin(config_cur[0]),  np.cos(config_cur[0]), 0, config_cur[2]],
                     [0, 0, 1, h],
                     [0, 0, 0, 1]])
    # find new chassis config
    T_cha_new = T_cha.dot(T_cha2cha_new)
    
    # find new robot config 
    # [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper state]
    config_new  = config_cur.copy()
    config_new[0:3] = np.array([arctan2(T_cha_new[1, 0], T_cha_new[0, 0]), T_cha_new[0, 3], T_cha_new[1, 3]])
    config_new[3:12] += np.array(vel) * dt
    
    return config_new
