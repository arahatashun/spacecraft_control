#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Shun Arahata
# This module visualize motion of a spin satellite using
# Matplotlib 3d plot refering to
# https://stackoverflow.com/questions/39666721/python-animating-a-vector-using-mplot3d-and-animation
import csv
from matplotlib import pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import Quaternion as qt

class Kinematic():
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,projection = '3d')
        self.fig.tight_layout()
        self.ax.view_init(40, -45)
        self.ax.set_xlim3d(-1,1)
        self.ax.set_ylim3d(-1,1)
        self.ax.set_zlim3d(-1,1)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.file = open('quaternion.csv', 'r')
        self.csv = csv.reader(file)
        line = next(self.csv)
        q0, q1, q2, q3 = line.strip(" \t\r\n[]").split(",")
        quat = float(q0), float(q1), float(q2), float(q3)

    def update(self):
        line = next(self.csv)

if __name__ == '__main__':
    main()
