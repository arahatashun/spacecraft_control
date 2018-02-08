#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Shun Arahata
# This module visualize motion of a spin satellite using
# Matplotlib 3d plot refering to
# https://stackoverflow.com/questions/39666721/python-animating-a-vector-using-mplot3d-and-animation
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np


def quat2dcm(q0, q1, q2, q3):
    # Quaternion to DCM
    x = np.array([q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2,
                2*(q1*q2 - q0*q3),
                2*(q1*q3 + q0*q2)])
    y = np.array([2*(q1*q2 + q0*q3),
                q0**2-q1**2 + q2**2-q3**2,
                2*(q2*q3 - q0*q1)])
    z = np.array([2*(q1*q3 - q0*q2),
                2*(q2*q3 + q0*q1),
                q0**2-q1**2-q2**2+q3**2])
    return x,y,z


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
        self.ax.scatter(0,0,0, color='k') # black origin dot
        self.file = open('quaternion.csv', 'r')
        self.csv = csv.reader(self.file)

    def setup_plot(self):
        line = next(self.csv)
        q0, q1, q2, q3 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
        x,y,z = quat2dcm(q0,q1,q2,q3)
        self.x_vec = self.ax.quiver(0, 0, 0, x[0], x[1], x[2], length=1, normalize=True,  color='r')
        self.y_vec = self.ax.quiver(0, 0, 0, y[0], y[1], y[2], length=1, normalize=True,  color='g')
        self.z_vec = self.ax.quiver(0, 0, 0, z[0], z[1], z[2], length=1, normalize=True,  color='b')
        return self.x_vec, self.y_vec, self.z_vec

    def update(self, i):
        line = next(self.csv)
        q0, q1, q2, q3 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
        x,y,z = quat2dcm(q0,q1,q2,q3)
        self.x_vec.remove()
        self.y_vec.remove()
        self.z_vec.remove()
        self.x_vec = self.ax.quiver(0, 0, 0, x[0], x[1], x[2], length=1, normalize=True,  color='r')
        self.y_vec = self.ax.quiver(0, 0, 0, y[0], y[1], y[2], length=1, normalize=True,  color='g')
        self.z_vec = self.ax.quiver(0, 0, 0, z[0], z[1], z[2], length=1, normalize=True,  color='b')
        return self.x_vec, self.y_vec, self.z_vec

    def make_animation(self):
         self.ani = animation.FuncAnimation(self.fig, self.update, init_func=self.setup_plot, blit=True, frames = 100)


if __name__ == '__main__':
    foo = Kinematic()
    foo.make_animation()
    foo.ani.save("Sample.gif", writer = 'imagemagick')
