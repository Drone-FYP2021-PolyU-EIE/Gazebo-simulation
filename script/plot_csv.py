#!/usr/bin/env python3
import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv

"""
This script can display the pointcloud which is stored in a csv file,
"""

# This program is not included in ROS
cloudXYZ = []
x,y,z = [],[],[]
ax = None

# Read pointcloud saved in csv
def read_csv_cloud(file_name):
    cloud = []
    with open(file_name) as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            cloud.append(row)  
    return cloud

def animate(i):
    global cloudXYZ,ax
    global x,y,z
    x.append(float(cloudXYZ[i][0]))
    y.append(float(cloudXYZ[i][1]))
    z.append(float(cloudXYZ[i][2]))
    plt.cla()
    print("index:{} x:{} y:{} z:{}".format(i,x[i],y[i],z[i]))
    # Plotting curve
    #ax.plot3D(x, y, z, color='blue',label='planned_path')
    # Plotting scatter points
    ax.scatter(x, y, z, z, cmap='jet', label='pointcloud')
    ax.set_title('Path Planner (m)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    plt.legend(loc='upper left')
    plt.tight_layout()

def main():
    global cloudXYZ,ax
    # Load the pointcloud
    cloudXYZ = read_csv_cloud('saved_point_cloud_XYZ.csv')
    x_s,y_s,z_s = [],[],[]

    # Plotter
    ax = plt.axes(projection='3d')
    ani = FuncAnimation(plt.gcf(), animate, interval=100)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    main()