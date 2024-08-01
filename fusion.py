import imufusion
import matplotlib.pyplot as pyplot
import numpy
import sys
import os
from pykalman import KalmanFilter

def datato_listarray(matrix,length):    
    for n in range(length):
        matrix[n*3] = [matrix[n*3], matrix[(n*3)+1], matrix[(n*3)+2]]
        matrix[(n*3)+1] = ''
        matrix[(n*3)+2] = ''
    matrix[:] = [x for x in matrix if x]
    for n in range(length):
        matrix[n] = numpy.array(list(map(float, matrix[n])))
    return matrix
       
def inertial(uncalibrated, misalignment, sensitivity, offset):
    return (numpy.matrix(misalignment) * numpy.diag(sensitivity) * numpy.array([uncalibrated - offset]).T).A1

def magnetic(uncalibrated, soft_iron_matrix, hard_iron_offset):
    return (numpy.matrix(soft_iron_matrix) * numpy.array([uncalibrated] - hard_iron_offset).T).A1

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

datadir_list = os.listdir("data")
datadir_list.reverse()

for n in datadir_list:
    dir_name = "data/" + n
    if os.path.isfile(dir_name + "/output/accel") and os.path.isfile(dir_name + "/output/gyro") and os.path.isfile(dir_name + "/output/mag"):
        accelerometerMisalignment, accelerometerOffset = get_matrix_vector(dir_name, "accel")        
        gyroscopeMisalignment, gyroscopeOffset = get_matrix_vector(dir_name, "gyro")
        softIronMatrix, hardIronOffset = get_matrix_vector(dir_name, "mag")


timestamp = data[:, 0]
accelerometer = data[:, 1:4]
gyroscope = data[:, 4:7]
magnetometer = data[:, 7:10]

gyroscopeSensitivity = [1, 1, 1]
accelerometerSensitivity = [1, 1, 1]
length = len(timestamp)

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


earth = numpy.empty((len(timestamp), 3))
euler = numpy.empty((len(timestamp), 3))
internal_states = numpy.empty((len(timestamp), 6))
flags = numpy.empty((len(timestamp), 4))

for index in range(len(timestamp)):
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


# def plot_bool(axis, x, y, label):
    # axis.plot(x, y, "tab:cyan", label=label)
    # pyplot.sca(axis)
    # pyplot.yticks([0, 1], ["False", "True"])
    # axis.grid()
    # axis.legend()


# Plot Euler angles
# figure, axes = pyplot.subplots(nrows=11, sharex=True, gridspec_kw={"height_ratios": [6, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]})

# figure.suptitle("Euler angles, internal states, and flags")

#axes[0].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
#axes[0].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
#axes[0].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
#axes[0].set_ylabel("Degrees")
#axes[0].grid()
#axes[0].legend()
#
## Plot initialising flag
#plot_bool(axes[1], timestamp, flags[:, 0], "Initialising")
#
## Plot angular rate recovery flag
#plot_bool(axes[2], timestamp, flags[:, 1], "Angular rate recovery")
#
## Plot acceleration rejection internal states and flag
#axes[3].plot(timestamp, internal_states[:, 0], "tab:olive", label="Acceleration error")
#axes[3].set_ylabel("Degrees")
#axes[3].grid()
#axes[3].legend()
#
#plot_bool(axes[4], timestamp, internal_states[:, 1], "Accelerometer ignored")
#
#axes[5].plot(timestamp, internal_states[:, 2], "tab:orange", label="Acceleration recovery trigger")
#axes[5].grid()
#axes[5].legend()
#
#plot_bool(axes[6], timestamp, flags[:, 2], "Acceleration recovery")
#
## Plot magnetic rejection internal states and flag
#axes[7].plot(timestamp, internal_states[:, 3], "tab:olive", label="Magnetic error")
#axes[7].set_ylabel("Degrees")
#axes[7].grid()
#axes[7].legend()
#
#plot_bool(axes[8], timestamp, internal_states[:, 4], "Magnetometer ignored")
#
#axes[9].plot(timestamp, internal_states[:, 5], "tab:orange", label="Magnetic recovery trigger")
#axes[9].grid()
#axes[9].legend()
#
#plot_bool(axes[10], timestamp, flags[:, 3], "Magnetic recovery")

import numpy as np
import pylab as pl
from pykalman import KalmanFilter

rot_pos = euler
axi = ["position_x","position_y","position_z"]
for n in range(len(axi)):
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
    print(earth)
    #earth = (np.delete(earth, 0, 0))
    print(earth)
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
    print('fitted model: {0}'.format(kf))
    
    # Plot lines for the observations without noise, the estimated position of the
    # target before fitting, and the estimated position after fitting.
    pl.figure(figsize=(16, 6))
    obs_scatter = pl.scatter(timestamp, observations, marker='x', color='b',
                             label='observations')
    position_line = pl.plot(timestamp, states_pred[:, -2],
                            linestyle='-', marker='o', color='r',
                            label='position est.')
    axi[n] = np.array([states_pred[:, -2]])
    print(axi[n])
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    print(b.shape)
    print(rot_pos.shape)
    print(axi[n].T.shape)
    rot_pos = np.concatenate((rot_pos, axi[n].T), axis=1)
    print(rot_pos)

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

# define the animation function
def rotate(rot_pos):
    ax.clear()

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    rot_pos = list(rot_pos)
    print(rot_pos)
    dtorads = np.pi*(1/180)
    # rotate the vertices around the x and y axes
    cx, sx = np.cos(dtorads*rot_pos[0]), np.sin(dtorads*rot_pos[0])
    cy, sy = np.cos(dtorads*rot_pos[1]), np.sin(dtorads*rot_pos[1])
    cz, sz = np.cos(dtorads*rot_pos[2]), np.sin(dtorads*rot_pos[2])
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
        lstvert[x][0] = lstvert[x][0] + rot_pos[3]
        lstvert[x][1] = lstvert[x][1] + rot_pos[4]
        lstvert[x][2] = lstvert[x][2] + rot_pos[5]
    vertices_pos = np.array(lstvert)
    vertices_rot = np.dot(rotz, (np.dot(roty, np.dot(rotx, vertices_pos.T)).T).T).T
    # plot the rotated cube
    for edge in edges:
        ax.plot3D(vertices_rot[edge, 0], vertices_rot[edge, 1], vertices_rot[edge, 2], 'blue')

# create the animation object
anim = animation.FuncAnimation(fig, rotate, frames=(rot_pos), interval=100)

# show the plot
plt.show()
