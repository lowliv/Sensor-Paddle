# echo-server.py

import os
import socket
import sys
import shutil
import time as TIME
from itertools import chain
from statistics import mean
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
    
def get_data(time, freq):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M")
    data_file = open("data/data-" + dt_string, 'w')
    data_file.writelines("time,accelx,accely,accelz,gyrox,gyroy,gyroz,magx,magy,magz\n")
    total = time*freq
    data_list = []
    new_entry = str(conn.recv(1024), "utf-8")
    while (new_entry != 'Ready for next instructions!' and valid_data(new_entry) == True):
        TIME.sleep(1/freq)
        data_list.append(new_entry)
        new_entry = str(conn.recv(1024), "utf-8")
        print(data_list)
            
    joined_data = ''.join(data_list)
    fixed_data = joined_data.replace('*', '\n')

    data_file.writelines(fixed_data)
    data_file.close()

    
def send_info():
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
        send_info()
    conn.sendall(bytes(str(freq) + "\n", "utf-8"))
    get_data(time,freq)
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
        
       


                  
                
