import os
import sys 
import csv
import time
import shutil

def calibrate(target_dir, sensor_type):
    if sensor_type == "accel":
        data_path = "/accdata"
    if sensor_type == "gyro":
        data_path = "/gyrodata"
    if sensor_type == "mag":
        data_path = "/magdata"
    PATH =  os.path.dirname(os.path.abspath(__file__))       
    with open(target_dir + data_path, newline='') as csvfile:
        datareader = csv.DictReader(csvfile)

        data = ""
        for row in datareader:
           data = data + row[sensor_type + 'x'] + ','
           data = data + row[sensor_type + 'y'] + ','
           data = data + row[sensor_type + 'z']
           data = data + '\n'

    os.chdir("Magnetometer-Calibration/ellipsoid_fitting")
    magdatafile = open("sensor_data.txt", "w")
    magdatafile.writelines(data)
    magdatafile.close()

    os.system("octave main_sensor_calibration.m > " + sensor_type)
    os.chdir(PATH)
    if not os.path.isdir(target_dir + "output"):
        os.mkdir(target_dir + "output")
    shutil.move("Magnetometer-Calibration/ellipsoid_fitting/" + sensor_type, target_dir + "output/" + sensor_type)
    
if (args_count := len(sys.argv)) > 2:
    print(f"One argument expected, got {args_count - 1}")
    raise SystemExit(2)
elif args_count < 2:
    print("You must specify the target directory")
    raise SystemExit(2)
if not os.path.isdir(str(sys.argv[1])):
    print("The target directory doesn't exist")
    raise SystemExit(1)
if not os.path.isfile(str(sys.argv[1]) + "accdata"):
    print ("The target directory is missing the accgyrodata file")
    raise SystemExit(1)
if not os.path.isfile(str(sys.argv[1]) + "gyrodata"):
    print ("The target directory is missing the accgyrodata file")
    raise SystemExit(1)
if not os.path.isfile(str(sys.argv[1]) + "magdata"):
    print ("The target directory is missing the magdata file")
    raise SystemExit(1)
    
target_dir = str(sys.argv[1])
sensor_type = ["accel", "gyro", "mag"]
for n in sensor_type:
    calibrate(target_dir, n)

