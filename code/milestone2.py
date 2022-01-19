import numpy as np
import modern_robotics as mr
import csv

"""[in order to generate the csv file, just change the directory in line 119 to your desired local directory]

line 119:   with open('/home/jameszhz//Desktop/ME449/final_project/ee_traj.csv', 'w') as f:
                writer = csv.writer(f)
                for row in final_traj:
                    writer.writerow(row)

This code write a trajectory generator function to generate a desired trajectory for the end-effecor to follow, 
given the initial configuration of the end effector, cube's initial configuration, cube's desired final configuration 
in space frame {s}; end-effector's configuration relative to the cube when it is grasping, end-effector's standoff configuration 
above the cube, before and after grasping, relative to the cube, and the number of trajectory reference configurations per 0.01 seconds.

"""

def TrajectoryGenerator(Tse_i, Tsc_i, Tsc_f, Tce_grasp, Tce_standoff, k):
    """
    Generates the trajectory for the end-effector frame {e} during eight segments
    :param Tse_i: Initial configuration of end effector
    :param Tsc_i: cube's initial configuration
    :param Tsc_f: cube's desired final configuration
    :param Tce_grasp: end-effector's configuration relative to the cube when it is grasping
    :param Tce_standoff: end-effector's standoff configuration above the cube, before and after grasping, relative to the cube
    :param k: number of trajectory reference configurations per 0.01 seconds
    """
    final_traj = []
    Tf = [4, 3, 3, 3, 6, 3, 3, 3] #total time of the motion in seconds from rest to rest
    method = 5 # time-scaling
    
    ####starting traj#####
    #Segment 1: move gripper from initial to standoff position 
    Xstart = Tse_i
    Xend = np.dot(Tsc_i, Tce_standoff)
    grip1 = 0
    gene_traj(Xstart, Xend, Tf[0], k, method, grip1, final_traj)
    
    #Segment 2: move gripper down to grasp position 
    Xstart = np.dot(Tsc_i, Tce_standoff)
    Xend = np.dot(Tsc_i, Tce_grasp)
    grip2 = 0
    gene_traj(Xstart, Xend, Tf[1], k, method, grip2, final_traj)
    
    #Segment 3: close gripper
    Xstart = np.dot(Tsc_i, Tce_grasp)
    Xend = np.dot(Tsc_i, Tce_grasp)
    grip3 = 1
    gene_traj(Xstart, Xend, Tf[2], k, method, grip3, final_traj)
    
    #Segment 4: move gripper up to the initial standoff position
    Xstart = np.dot(Tsc_i, Tce_grasp)
    Xend = np.dot(Tsc_i, Tce_standoff)
    grip4 = 1
    gene_traj(Xstart, Xend, Tf[3], k, method, grip4, final_traj)
    
    #Segment 5: move gripper from initial standoff to final standoff position
    Xstart = np.dot(Tsc_i, Tce_standoff)
    Xend = np.dot(Tsc_f, Tce_standoff)
    grip5 = 1
    gene_traj(Xstart, Xend, Tf[4], k, method, grip5, final_traj)   
    
    #Segment 6: move gripper down to grasp position
    Xstart = np.dot(Tsc_f, Tce_standoff)
    Xend = np.dot(Tsc_f, Tce_grasp)
    grip6 = 1
    gene_traj(Xstart, Xend, Tf[5], k, method, grip6, final_traj)      
    
    #Segment 7: open gripper
    Xstart = np.dot(Tsc_f, Tce_grasp)
    Xend = np.dot(Tsc_f, Tce_grasp)
    grip7 = 0
    gene_traj(Xstart, Xend, Tf[6], k, method, grip7, final_traj)        
    
    #segment 8: move gripper up to the final standoff position 
    Xstart = np.dot(Tsc_f, Tce_grasp)
    Xend = np.dot(Tsc_f, Tce_standoff)
    grip8 = 0
    gene_traj(Xstart, Xend, Tf[7], k, method, grip8, final_traj)    
    
    
    return final_traj  

   
def gene_traj(Xstart, Xend, Tf, k, method, grip, final_traj):
    N = (Tf * k) / 0.01 #The number of points N > 1 (Start and stop) in the discrete representation of the trajectory
    traj = mr.ScrewTrajectory(Xstart, Xend, Tf, N, method)
    for t in traj:
        final_traj.append([t[0,0], t[0,1], t[0,2], t[1,0], t[1,1], t[1,2], t[2,0], t[2,1], t[2,2], t[0,3], t[1,3], t[2,3], grip])
        
