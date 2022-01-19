import numpy as np
import modern_robotics as mr
import csv
from modern_robotics import MatrixExp6, VecTose3, se3ToVec, TransInv, MatrixLog6, Adjoint, FKinBody, JacobianBody

def FeedbackControl(config, X_d, X_d_next, Kp, Ki, dt, errors=np.zeros(6), tolerance=1e-4):
    """A simulator for the kinematics
    Inputs
    :param config: The current actual end-effector configuration X, A 8-vector: 
        [chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5], need to convert to X (T_se)
    :param X_d: The current end-effector reference configuration, T_se_d
    :param X_d_next: The end-effector reference configuration at the next timestep in the reference trajectory at dt later, T_se_d_next
    :param Kp: Gain matrix I
    :param Ki: Gain matrix I 
    :param dt: timestep 
    
    Output
    :param Vt : The commanded end-effector twist V expressed in the end-effector frame {e}.
    """
    # car parameters
    r = 0.0475 # radius of wheel
    l = 0.47 / 2 # half length
    w = 0.3 / 2 # half wide
    h = 0.0963 # height
    
    #The fixed offset from the chassis frame {b} to the base frame of the arm {0}
    T_b0 = np.array([[1, 0, 0, 0.1662],
                    [0, 1, 0, 0], 
                    [0, 0, 1, 0.0026], 
                    [0, 0, 0, 1]])
    
    #The fixed offset from the chassis frame {b} to the base frame of the arm {0}
    M_0e = np.array([[1, 0, 0,  0.033],
                    [0, 1, 0, 0], 
                    [0, 0, 1, 0.6546], 
                    [0, 0, 0, 1]])
    #screw axes 
    B_list = np.array([[0,  0,  1, 0, 0.033, 0], 
                       [0, -1,  0, -0.5076, 0, 0], 
                       [0, -1,  0, -0.3526, 0, 0], 
                       [0, -1,  0, -0.2176, 0, 0], 
                       [0,  0,  1, 0, 0, 0]]).T
    
    #find pseudo-inverse for chassis
    H_pse = r / 4 * np.array([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [-1/(l+w), 1/(l+w), 1/(l+w), -1/(l+w)],
                              [1, 1, 1, 1],
                              [-1, 1, -1, 1],
                              [0, 0, 0, 0]])
    #function to find the T_se from a 12-vector input 
    def Get_X(config):
        arm_joint_angles = config[3:8]
        T_0e = mr.FKinBody(M_0e, B_list, arm_joint_angles)
        T_sb = np.array([[np.cos(config[0]), -np.sin(config[0]), 0, config[1]],
                        [np.sin(config[0]), np.cos(config[0]), 0, config[2]], 
                        [0, 0, 1, h], 
                        [0, 0, 0, 1]])
        return T_sb.dot(T_b0).dot(T_0e)
    
    ## find kinematic task-space feedforward FF 
    T_d2d_next = TransInv(X_d).dot(X_d_next)
    V_d = se3ToVec(1/dt * MatrixLog6(T_d2d_next))
    # FF = Ad(X^-1Xd)*Vd
    T_err = TransInv(Get_X(config)).dot(X_d)
    FF = Adjoint(T_err).dot(V_d)
    ##
    
    # calculate errors (this errors also accumulate outside)
    X_err = se3ToVec(MatrixLog6(T_err))
    errors += dt * X_err
    
    # find Vt
    Vt = FF + Kp.dot(X_err) + Ki.dot(errors)
    
    # find J_base 
    T_0e = FKinBody(M_0e, B_list, config[3:8])
    T_eb = TransInv(T_0e).dot(mr.TransInv(T_b0))
    J_base = Adjoint(T_eb).dot(H_pse)
    
    # find J_arm
    J_arm = JacobianBody(B_list, config[3:8])
    
    # find mobile manipulator jacobian J_e
    J_e = np.concatenate((J_base, J_arm), axis=1)
    
    # find joint velocity [u (4), thetadot (5)] from pseudo-inverse
    J_e_pse = np.linalg.pinv(J_e, tolerance) # tolerance specify how close to zero a singular value must be to be treated as zero.
    J_vel = J_e_pse.dot(Vt)   
       
    return Vt, J_vel, X_err