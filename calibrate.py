import os
import sys 
import csv
import time

if (args_count := len(sys.argv)) > 2:
    print(f"One argument expected, got {args_count - 1}")
    raise SystemExit(2)
elif args_count < 2:
    print("You must specify the target directory")
    raise SystemExit(2)
if not os.path.isfile(str(sys.argv[1])):
    print("The target file doesn't exist")
    raise SystemExit(1)
target_file = str(sys.argv[1])

# This isolates the mag sensor values
with open(target_file, newline='') as csvfile:
    datareader = csv.DictReader(csvfile)
    
    magdata = ""
    for row in datareader:
       magdata = magdata + row['magx'] + ','
       magdata = magdata + row['magy'] + ','
       magdata = magdata + row['magz']
       magdata = magdata + '\n'

os.chdir("Magnetometer-Calibration/ellipsoid_fitting")
magdatafile = open("sensor_data.txt", "w+")
magdatafile.writelines(magdata)
magdatafile.close()

os.system("octave main_sensor_calibration.m > magcal")
