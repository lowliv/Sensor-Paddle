import imufusion
import matplotlib.pyplot as pyplot
import numpy
import sys
import os
from pykalman import KalmanFilter
import time

if (args_count := len(sys.argv)) > 2:
    print(f"One argument expected, got {args_count - 1}")
    raise SystemExit(2)
elif args_count < 2:
    print("You must specify the target file")
    raise SystemExit(2)
if not os.path.isfile(str(sys.argv[1])):
    print("The target directory doesn't exist")
    raise SystemExit(1)

target_file = str(sys.argv[1])

data = numpy.genfromtxt(target_file, delimiter=",", skip_header=1)


sample_rate = 100  # 100 Hz

def sortfunct(n):
    return int(str(n[slice(23,27)])+str(n[slice(20,22)])+str(n[slice(17,19)])+str(n[slice(28,30)])+str(n[slice(31,33)]))
    
datadir_list = os.listdir("data")
for n in datadir_list:
    if "calibration" not in n:
        datadir_list[:] = ["" if x==n else x for x in datadir_list]

datadir_list[:] = [x for x in datadir_list if x]        
datadir_list.sort(key=sortfunct,reverse=True)

def get_matrix_vector(dir_name, sensor_type):
    data_file = open(dir_name + "/output/" + sensor_type)
    data_list = data_file.readlines()
    matrix = data_list[3].split(' ') + data_list[4].split(' ') + data_list[5].split(' ')
    for n in range(len(matrix)):
        matrix[n] = matrix[n].replace('\n', '')
        matrix[n] = matrix[n].replace(' ', '')
    matrix[:] = [x for x in matrix if x]
    matrix = datato_listarray(matrix,3)
    vector = data_list[8].split('\n') + data_list[9].split('\n') + data_list[10].split('\n')
    for n in range(len(vector)):
        vector[n] = vector[n].replace('\n', '')
        vector[n] = vector[n].replace(' ',' ')
    vector[:] = [x for x in vector if x]
    vector = numpy.array(list(map(float, vector)))
    return matrix, vector
     
def inertial(uncalibrated, misalignment, sensitivity, offset):
    return (numpy.matrix(misalignment) * numpy.diag(sensitivity) * numpy.array([uncalibrated - offset]).T).A1

def magnetic(uncalibrated, soft_iron_matrix, hard_iron_offset):
    return (numpy.matrix(soft_iron_matrix) * numpy.array([uncalibrated] - hard_iron_offset).T).A1

def datato_listarray(matrix,length):    
    for n in range(length):
        matrix[n*3] = [matrix[n*3], matrix[(n*3)+1], matrix[(n*3)+2]]
        matrix[(n*3)+1] = ''
        matrix[(n*3)+2] = ''
    matrix[:] = [x for x in matrix if x]
    for n in range(length):
        matrix[n] = numpy.array(list(map(float, matrix[n])))
    return matrix
    
for n in datadir_list:
    dir_name = "data/" + n
    if os.path.isfile(dir_name + "/output/accel") and os.path.isfile(dir_name + "/output/gyro") and os.path.isfile(dir_name + "/output/mag"):
        accelerometerMisalignment, accelerometerOffset = get_matrix_vector(dir_name, "accel")        
        gyroscopeMisalignment, gyroscopeOffset = get_matrix_vector(dir_name, "gyro")
        softIronMatrix, hardIronOffset = get_matrix_vector(dir_name, "mag")
        break        

timestamp = data[:, 0]
accelerometer = data[:, 1:4]
gyroscope = data[:, 4:7]
magnetometer = data[:, 7:10]

gyroscopeSensitivity = [1, 1, 1]
accelerometerSensitivity = [1, 1, 1]
length = len(timestamp)
freq = timestamp[1] - timestamp[0]

gyroscope = inertial(gyroscope, gyroscopeMisalignment, gyroscopeSensitivity, gyroscopeOffset)
accelerometer = inertial(accelerometer, accelerometerMisalignment, accelerometerSensitivity, accelerometerOffset)
magnetometer = magnetic(magnetometer, softIronMatrix, hardIronOffset)
gyroscope = gyroscope.tolist()
accelerometer = accelerometer.tolist()
magnetometer = magnetometer.tolist()
gyroscope = datato_listarray(gyroscope,length)
accelerometer = datato_listarray(accelerometer,length)
magnetometer = datato_listarray(magnetometer,length)
            
# Instantiate algorithms
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,  # convention
                                   0.5,  # gain
                                   2000,  # gyroscope range
                                   10,  # acceleration rejection
                                   10,  # magnetic rejection
                                   5 * sample_rate)  # recovery trigger period = 5 seconds

# Process sensor data
delta_time = numpy.diff(timestamp, prepend=timestamp[0])


earth = numpy.empty((length, 3))
euler = numpy.empty((length, 3))
internal_states = numpy.empty((length, 6))
flags = numpy.empty((length, 4))

for index in range(length):
    gyroscope[index] = offset.update(gyroscope[index])

    ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index], delta_time[index])

    euler[index] = ahrs.quaternion.to_euler()
    earth[index] = ahrs.earth_acceleration
    
    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = numpy.array([ahrs_internal_states.acceleration_error,
                                          ahrs_internal_states.accelerometer_ignored,
                                          ahrs_internal_states.acceleration_recovery_trigger,
                                          ahrs_internal_states.magnetic_error,
                                          ahrs_internal_states.magnetometer_ignored,
                                          ahrs_internal_states.magnetic_recovery_trigger])

    ahrs_flags = ahrs.flags
    flags[index] = numpy.array([ahrs_flags.initialising,
                                ahrs_flags.angular_rate_recovery,
                                ahrs_flags.acceleration_recovery,
                                ahrs_flags.magnetic_recovery])


import numpy as np
import pylab as pl
from pykalman import KalmanFilter

rot_pos = euler
pos= []
for n in range(3):
    dt = timestamp[1] - timestamp[0]
    Acc_Variance = 0.00007
    # transition_matrix  
    F = [[1, dt, 0.5*dt**2], 
         [0,  1, dt],
         [0,  0, 1]]
    # observation_matrix   
    H = [0, 0, 1]
    
    # transition_covariance 
    Q = [[0.2,    0,      0], 
         [  0,  0.1,      0],
         [  0,    0,  10e-4]]
    
    # observation_covariance 
    R = Acc_Variance
    
    # initial_state_mean
    X0 = [0,
          0,
          earth[1][n]]
    
    # initial_state_covariance
    P0 = [[  0,    0,               0], 
          [  0,    0,               0],
          [  0,    0,   Acc_Variance]]

    n_timesteps = length
    #earth = (np.delete(earth, 0, 0))
    observations = [value[n] for value in earth] 
    # create a Kalman Filter by hinting at the size of the state and observation
    # space.  If you already have good guesses for the initial parameters, put them
    # in here.  The Kalman Filter will try to learn the values of all variables.
    kf = KalmanFilter(transition_matrices=F,
                      transition_covariance=Q,
                      initial_state_mean=X0)
    
    # You can use the Kalman Filter immediately without fitting, but its estimates
    # may not be as good as if you fit first.
    states_pred = kf.em(observations).smooth(observations)[0]
    
    # Plot lines for the observations without noise, the estimated position of the
    # target before fitting, and the estimated position after fitting.
    pl.figure(figsize=(16, 6))
    obs_scatter = pl.scatter(timestamp, observations, marker='x', color='b',
                             label='observations')
    position_line = pl.plot(timestamp, states_pred[:, -2],
                            linestyle='-', marker='o', color='r',
                            label='position est.')
    pos.append(np.array([states_pred[:, -2]]))
    rot_pos = np.concatenate((rot_pos, pos[n].T), axis=1)

    velocity_line = pl.plot(timestamp, states_pred[:, -1],
                            linestyle='-', marker='o', color='g',
                            label='velocity est.')
    acceleration_line = pl.plot(timestamp, states_pred[:,0],
                                linestyle='-', marker='o', color='b',
                                label='acceleration est.')
    pl.legend(loc='lower right')
    pl.xlim(xmin=0, xmax=timestamp.max())
    pl.xlabel('time')
    pl.show()

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# define cube vertices
vertices = np.array([[-.1, -.1, -.1],
                     [.3, -.1, -.1],
                     [.3, 0, -.1],
                     [-.1, 0, -.1],
                     [-.1, -.1, .1],
                     [.3, -.1, .1],
                     [.3, 0, .1],
                     [-.1, 0, .1]])

# define edges joining the vertices
edges = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

# define the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#set plot limits
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

global speed_mult
speed_mult = 2
#def speed_up
#    speed_mult = speed_mult+.1
def keyactions(event):
    global speed_mult
    if event.key == '+':
        speed_mult -= 0.1
    if event.key == '-':
        speed_mult += 0.1
vertices_rot = []
for n in range(len(rot_pos)):
    rot_pos[n] = list(rot_pos[n])
    dtorads = np.pi*(1/180)
    # rotate the vertices around the x,y,z axes
    cx, sx = np.cos(dtorads*rot_pos[n][0]), np.sin(dtorads*rot_pos[n][0])
    cy, sy = np.cos(dtorads*rot_pos[n][1]), np.sin(dtorads*rot_pos[n][1])
    cz, sz = np.cos(dtorads*rot_pos[n][2]), np.sin(dtorads*rot_pos[n][2])
    rotx = np.array([[1, 0, 0],
                     [0, cx, -sx],
                     [0, sx, cx]])
    roty = np.array([[cy, 0, sy],
                     [0, 1, 0],
                     [-sy, 0, cy]])
    rotz = np.array([[cz, -sz, 0],
                     [sz, cz, 0],
                     [0, 0, 1]])
    lstvert = np.ndarray.tolist(vertices)
    for x in range(len(lstvert)):
        lstvert[x][0] = lstvert[x][0] + rot_pos[n][3]
        lstvert[x][1] = lstvert[x][1] + rot_pos[n][4]
        lstvert[x][2] = lstvert[x][2] + rot_pos[n][5]
    vertices_pos = np.array(lstvert)
    vertices_rot.append(np.dot(rotz, (np.dot(roty, np.dot(rotx, vertices_pos.T)).T).T).T)
    print(vertices_rot)
# define the animation function
def update(vertices_rot):
    start = time.perf_counter()
    ax.clear()
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    for edge in edges:
        ax.plot3D(vertices_rot[edge, 0], vertices_rot[edge, 1], vertices_rot[edge, 2], 'blue')
    cid = fig.canvas.mpl_connect('key_press_event', keyactions)
    stop = time.perf_counter()
    print(stop-start)
    print(freq*speed_mult)
    if freq*speed_mult > stop-start:
        time.sleep((freq*speed_mult)-(stop-start))
    else:
        print("Frame was unable to be displayed in time")
    
# create the animation object
anim = animation.FuncAnimation(fig, update, frames=(vertices_rot), interval=0)

# show the plot
plt.show()
