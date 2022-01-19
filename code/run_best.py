import modern_robotics as mr
from milestone1 import NextState
from milestone2 import TrajectoryGenerator
from milestone3 import FeedbackControl
import numpy as np
import matplotlib.pyplot as plt
import logging
import csv

""" This file imports all needed functions from milestones and runs in the main function Simulate.

    To Run:
    In your /code directory, run python3 run_best.py
    
    Retunrs all files (log file, Xerr csv, Traj_file csv, image) in the same directory
    If you want to save in different directory, may simply change the path in line 82, 88, 95, 112 respectively 
"""

def Simulate(config_i, Tse_i, Tsc_i, Tsc_f, Kp, Ki, dt, v_lim, log_file, Traj_file, Xerr_file, img):
    """
    Input:
    :param config_i: Inital configuration of a cube
    :param Tse_i: Inital configuration of the end_effector 
    :param Tsc_i: Inital configuration of a cube 
    :param Tsc_f: Final configuration of a cube
    :param dt: time step
    :param v_lim: Speed limit for joints speed
    :param Kp: P Gains
    :param Ki: I Gains
    :param log_file: Output path for log file
    :param Traj_file: Output path for log file
    :param Xerr_file: Output path for log file
    Output:
    ee_traj: List of N x 13 trajectory list
    Xerr_list: List of N x 6 error list
    files:
            log file
            Xerr csv 
            Traj_file csv
            image
    """
    ee_traj = []
    Xerr_list = []
    
    Tce_grasp = np.array([[-np.sqrt(2)/2,0,np.sqrt(2)/2,0],
                [0,1,0,0],
                [-np.sqrt(2)/2,0,-np.sqrt(2)/2,0],
                [0,0,0,1]])

    Tce_standoff = np.array([[-np.sqrt(2)/2,0,np.sqrt(2)/2,0],
                      [0,1,0,0],
                      [-np.sqrt(2)/2,0,-np.sqrt(2)/2,0.1],
                      [0,0,0,1]])
    k = 1
    # generate traj
    traj = TrajectoryGenerator(Tse_i, Tsc_i, Tsc_f, Tce_grasp, Tce_standoff, k)
    N = len(traj)
    # convert traj to list of tuple [Tse, gripper state t[12]]
    T_traj = [(np.array([[t[0], t[1], t[2], t[9]],
                        [t[3], t[4], t[5], t[10]],
                        [t[6], t[7], t[8], t[11]],
                        [0,    0,    0,    1]]),   t[12])
                        for t in traj]
    errors = np.zeros(6)
    # start loop iteration 
    config = config_i.copy()
    ee_traj.append(config.tolist())
    for i in range(N - 1):  # iter N - 1
        ee_traj.append(config.tolist())
        # find commanded end-effector twist V, joint_vel [u, thetadot], X_err
        Vt, J_vel, X_err = FeedbackControl(config, T_traj[i][0], T_traj[i+1][0], Kp, Ki, dt, errors, tolerance=1e-4)
        Xerr_list.append(X_err.tolist())
        # find 9-vector of controls indicating the arm joint speeds, be careful for the order of input
        v_ctrl = np.concatenate((J_vel[4:9], J_vel[0:4]))
        # find next config
        config = NextState(config, v_ctrl, dt, v_lim)
        config[12] = T_traj[i][1]        
    
    # write log file
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    # Write the traj csv file
    logging.debug("Generating animation csv file.")
    with open(Traj_file, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in ee_traj:
            csv_writer.writerow(row)

    # Write the Xerr csv file
    logging.debug("Writing error plot data.")
    with open(Xerr_file, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in Xerr_list:
            csv_writer.writerow(row)

    # Plot error
    x = np.linspace(0, 10, len(Xerr_list))
    y = np.asarray(Xerr_list)
    for i in range(6):
        plt.plot(x, y.T[i])

    plt.title("Plot of Xerr in SE3")
    plt.xlabel("Time in sec")
    plt.ylabel("Xerr in rad or m")
    plt.legend([r'$X_{err}[1]$', r'$X_{err}[2]$',
                r'$X_{err}[3]$', r'$X_{err}[4]$',
                r'$X_{err}[5]$', r'$X_{err}[6]$'])
    plt.savefig(img)
    plt.show()

    logging.debug("Done.")       

if __name__ == "__main__":
    #  [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5]
    config_i = np.array([-0.8, 0.2, 0.2, 0., 0., 0.2, -1.6, 0., 0., 0., 0., 0., 0.])
    
    Tse_i = np.array([[0,0,1,0],
                    [0,1,0,0],
                    [-1,0,0,0.5],
                    [0,0,0,1]])
    
    dt = 0.01
    v_lim = 12.3
    
    #best case
    Tsc_i = np.array([[1,0,0,1],
                      [0,1,0,0],
                      [0,0,1,0.025],
                      [0,0,0,1]])

    Tsc_f = np.array([[0,1,0,0],
                      [-1,0,0,-1],
                      [0,0,1,0.025],
                      [0,0,0,1]])
    Kp = np.eye(6) * 8
    Ki = np.eye(6) * 0.5
    print('best')
    Simulate(config_i, Tse_i, Tsc_i, Tsc_f, Kp, Ki, dt, v_lim, log_file='best.log', Traj_file='best_traj.csv', Xerr_file='best_Xerr.csv', img ='best.png')