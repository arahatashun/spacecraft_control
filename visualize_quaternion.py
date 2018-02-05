#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Shun Arahata

import bpy
from mathutils import Vector, Quaternion


def main():
    satellite = bpy.context.object # active object, assuming there is one
    satellite.rotation_mode = 'QUATERNION'
    with open('quaternion.csv', 'r') as file:
        for i, line in enumerate(file):
            q0, q1, q2, q3 = line.strip(" \t\r\n[]").split(",")
            loc = float(0), float(0), float(0)
            quat = float(q0), float(q1), float(q2), float(q3)
            satellite.location = loc
            satellite.rotation_quaternion = quat
            satellite.keyframe_insert('location', frame=i)
            satellite.keyframe_insert('rotation_quaternion', frame=i)

if __name__ == '__main__':
    main()
