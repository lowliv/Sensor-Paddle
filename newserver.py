# echo-server.py

import os
import socket
import sys
import time as TIME
from itertools import chain
from statistics import mean
from datetime import datetime

HOST = "192.168.1.17"  # Standard loopback interface address (localhost)
PORT = 60321  # Port to listen on (non-privileged ports are > 1023)

avg_accelX = 0
avg_accelY = 0
avg_accelZ = 0

avg_gyroX = 0
avg_gyroY = 0
avg_gyroZ = 0

avg_magX = 0
avg_magY = 0
avg_magZ = 0

def valid_data(n):
    try:
        float(n)
        return True
    except ValueError:
        if '*' in n:
            return True
        return False
    
def load_calibration():
    calibration_file = open("calibration", 'r')
    calibration_list = list(calibration_file)
    if (len(calibration_list) != 9):
        return False
    global avg_accelX, avg_accelY, avg_accelZ, avg_gyroX, avg_gyroY, avg_gyroZ, avg_magX, avg_magY, avg_magZ
    avg_accelX = float(calibration_list[0])
    avg_accelY = float(calibration_list[1])
    avg_accelZ = float(calibration_list[2])

    avg_gyroX = float(calibration_list[3])
    avg_gyroY = float(calibration_list[4])
    avg_gyroZ = float(calibration_list[5])

    avg_magX = float(calibration_list[6])
    avg_magY = float(calibration_list[7])
    avg_magZ = float(calibration_list[8])
    calibration_file.close()

def calibrate():
    if (os.path.isfile("calibration")):
        os.remove("calibration")
    calibration_file = open("calibration", 'w')
    data_list = []
    new_entry = str(conn.recv(1024), "utf-8")
    while (new_entry != 'Ready for next instructions!' and valid_data(new_entry) == True):
        TIME.sleep(1)
        data_list.append(new_entry)
        new_entry = str(conn.recv(1024), "utf-8")
        
    joined_data = ''.join(data_list)
    split_data_strings = joined_data.split("*")
    del split_data_strings[len(split_data_strings) - 1]
    print(split_data_strings)
    
    split_data = list(map(float, split_data_strings))
    print(split_data)
    enter = "\n"
    calibration_file.writelines(str(mean(split_data[::9])) + enter + str(mean(split_data[1::9])) + enter + str(mean(split_data[2::9])) + enter + str(mean(split_data[3::9])) + enter + str(mean(split_data[4::9])) + enter + str(mean(split_data[5::9])) + enter + str(mean(split_data[6::9])) + enter + str(mean(split_data[7::9])) + enter + str(mean(split_data[8::9])))
    calibration_file.close()
    
def write_file(time, freq):
    load_calibration()
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M")
    data_file = open("data/data-" + dt_string, 'w')
    total = time*freq
    data_list = []
    new_entry = str(conn.recv(1024), "utf-8")
    while (new_entry != 'Ready for next instructions!' and valid_data(new_entry) == True):
        TIME.sleep(1/freq)
        data_list.append(new_entry)
        new_entry = str(conn.recv(1024), "utf-8")
        print(data_list)
            
    joined_data = ''.join(data_list)
    split_data_strings = joined_data.split("*")
    del split_data_strings[len(split_data_strings)-1]
    split_data = list(map(float, split_data_strings))
    
    accelX = split_data[::9]
    accelX = [str(round(x - avg_accelX, 2)) for x in accelX]
    accelY = split_data[1::9]
    accelY = [str(round(x - avg_accelY, 2)) for x in accelY]
    accelZ = split_data[2::9]
    accelZ = [str(round(x - avg_accelZ, 2)) for x in accelZ]
    gyroX  = split_data[3::9]
    gyroX = [str(round(x - avg_gyroX, 2)) for x in gyroX]
    gyroY = split_data[4::9]
    gyroY = [str(round(x - avg_gyroY, 2)) for x in gyroY]
    gyroZ = split_data[5::9]
    gyroZ = [str(round(x - avg_gyroZ, 2)) for x in gyroZ]
    magX = split_data[6::9]
    magX = [str(round(x - avg_magX, 2)) for x in magX]
    magY = split_data[7::9]
    magY = [str(round(x - avg_magY, 2)) for x in magY]
    magZ = split_data[8::9]
    magZ = [str(round(x - avg_magZ, 2)) for x in magZ]
    print(accelX)
    print(str(accelX))
    
    enter = ["\n"]*total
    space = [" "]*total    
    data_file.writelines(list(chain.from_iterable(zip(accelX, space, accelY, space, accelZ, enter, gyroX, space, gyroY, space, gyroZ, enter, magX, space, magY, space, magZ, enter))))
    data_file.close()
    
def send_info():
    time = int(input("How many seconds should the data be recorded for (0 for calibration)?\nInput:"))
    if (time != 0 and load_calibration() != False):
        confirm = input("Are you sure you want to record data without calibrating?\nY/N:")    
        if (confirm == "Y" or confirm == "y"):
            conn.sendall(bytes(str(time) + "\n", "utf-8"))
        else:
            send_info()
    else:    
        if (time == 0):
            conn.sendall(bytes(str(time) + "\n", "utf-8"))
            calibrate()
            send_info()
        conn.sendall(bytes(str(time) + "\n", "utf-8"))
    status = str(conn.recv(1024), "utf-8")
    print(status)
    freq = int(input("How many data points per second should be recorded?\nInput:"))
    conn.sendall(bytes(str(freq) + "\n", "utf-8"))
    write_file(time,freq)
    send_info()

def get_status():
    print(f"Connected by {addr}")
    status = str(conn.recv(1024), "utf-8")
    print(status)
    if status == "LSM6DS and LIS3MDL Found! Waiting for commands.":
        with conn:
            send_info()
    
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        get_status()
        
       


                  
                
