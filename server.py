# echo-server.py

import os
import socket
import shutil
import time as TIME
from datetime import datetime

HOST = "192.168.1.17"  # Standard loopback interface address (localhost)
PORT = 60321  # Port to listen on (non-privileged ports are > 1023)

def valid_data(n):
    try:
        float(n)
        return True
    except ValueError:
        if ',' in n:
            return True
        if '*' in n:
            return True
        return False
    
def get_data(time, freq, datafile, label):
    data_file = open(datafile, 'a')
    if label == 1:
        data_file.writelines("time,accelx,accely,accelz,gyrox,gyroy,gyroz,magx,magy,magz\n")
    data_list = []
    new_entry = str(conn.recv(1024), "utf-8")
    while (new_entry != 'Ready for next instructions!' and valid_data(new_entry) == True):
        TIME.sleep(1/freq)
        time = time-(1/freq)
        print(time)
        data_list.append(new_entry)
        new_entry = str(conn.recv(1024), "utf-8")
            
    joined_data = ''.join(data_list)
    fixed_data = joined_data.replace('*', '\n')

    data_file.writelines(fixed_data)
    data_file.close()
    
def calibrate(datadir):
    os.mkdir(datadir)
    print ("Calibrating accelerometer and gyroscope. Keep the device completely still in one orientation and rotate it to a new orientation")
    count = 0
    time = 1
    freq = 5
    while count < 6:
        input("Is the device still? (Enter anything when ready)\n")
        conn.sendall(bytes(str(time) + "\n", "utf-8"))
        status = str(conn.recv(1024), "utf-8")
        conn.sendall(bytes(str(freq) + "\n", "utf-8"))
        count = count + 1
        get_data(time,freq,datadir + "/accgyrodata", count)
        
    input("Calibrating magnetometer. Rotate the device into as many orientations as possible. (Enter anything when ready)\n")
    time = 15
    freq = 5
    conn.sendall(bytes(str(time) + "\n", "utf-8"))
    status = str(conn.recv(1024), "utf-8")
    conn.sendall(bytes(str(freq) + "\n", "utf-8"))
    get_data(time,freq,datadir + "/magdata", 1)

    
def send_info():
    calibrate_response = input("Do you want to calibrate the sensors?\nInput:")
    response_list = ["yes", "Yes", "y", "Y"]
    for n in response_list:
        if calibrate_response == n:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H:%M")
            datadir = "data/calibration_data-" + dt_string
            calibrate(datadir)
    time = int(input("How many seconds should the data be recorded?\nInput:"))
    if time == 0:
        print("Enter a bigger number")
        send_info()
    conn.sendall(bytes(str(time) + "\n", "utf-8"))
    status = str(conn.recv(1024), "utf-8")
    print(status)
    freq = int(input("How many data points per second should be recorded?\nInput:"))
    if freq == 0:
        print("Enter a bigger number")
        while freq == 0:
            freq = int(input("How many data points per second should be recorded?\nInput:"))
    conn.sendall(bytes(str(freq) + "\n", "utf-8"))
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M")
    datafile = "data/data" + data_string
    label = 1
    get_data(time,freq,datafile,label)
    send_info()

def get_status():
    print(f"Connected by {addr}")
    status = str(conn.recv(1024), "utf-8")
    print(status)
    if status == "LSM6DS and LIS3MDL Found! Waiting for commands.":
        send_info()
    
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        get_status()
        
       


                  
                
