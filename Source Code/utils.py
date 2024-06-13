import os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from kinem import *

def cleanWrite():
    dirvec = os.listdir()
    if "Step Files" in dirvec:
        try:
            os.rmdir("Step Files")
        except:
            print("ERROR: Unable to reset 'Step Files' directory")
    else:
        dirvec = os.listdir()
        dirresults = list(map(lambda x: float(x) if x.replace('.', '', 1).isdigit() else 0.0, dirvec))
        for i in range(len(dirresults)):
            os.removedirs(str(dirresults[i]))
        # os.remove("*~")

def simple_interp(x1, x2, y1, y2, x):
    y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return y

def find_tstep_cos(kin):
    dtstar = 15.0
    dt_tmp = 0.015 * 0.2 / (kin.k * kin.amp)
    dtstar = min([dt_tmp, dtstar])
    return dtstar

def find_tstep_sin(kin):
    dtstar = 15.0
    dt_tmp = 0.015 * 0.2 / (kin.k * kin.amp)
    dtstar = min([dt_tmp, dtstar])
    return dtstar

def find_tstep_eldup(kin):
    dtstar = 1.0
    dtstar = min([0.015 * 0.2 / kin.K, 0.015])
    return dtstar

def find_tstep_elduptstart(kin):
    dtstar = 1.0
    dtstar = min([0.015 * 0.2 / kin.K, 0.015])
    return dtstar

def find_tstep_eldrampreturn(kin):
    dtstar = 1.0
    dtstar = min([0.015 * 0.2 / kin.K, 0.015])
    return dtstar

def find_tstep_bending(kin):
    dtstar = 15.0
    amp = kin.spl(kin.spl.t[-1])
    dt_tmp = 0.015 * 0.2 / (kin.k * amp)
    dtstar = min([dt_tmp, dtstar])
    return dtstar

def find_tstep_const(kin):
    dtstar = 0.015
    return dtstar

def simpleTrapz(y, x):
    if len(y) != len(x):
        raise ValueError("Vectors must be of the same length")
    r = 0.0
    for i in range(1, len(y)):
        r += (x[i] - x[i-1]) * (y[i] + y[i-1])
    return r / 2.0

def camber_calc(x, airfoil):
    # Determine camber and camber slope on airfoil from airfoil input file

    ndiv = len(x)
    c = x[-1]

    cam = np.zeros(ndiv)
    cam_slope = np.zeros(ndiv)
    
    in_air = np.loadtxt(airfoil)
    xcoord = in_air[:, 0]
    ycoord = in_air[:, 1]
    ncoord = len(xcoord)
    xcoord_sum = np.zeros(ncoord)

    for i in range(1, ncoord):
        xcoord_sum[i] = xcoord_sum[i-1] + abs(xcoord[i] - xcoord[i-1])

    y_spl = interp1d(xcoord_sum, ycoord, kind='linear', fill_value='extrapolate')

    y_ans = np.zeros(2 * ndiv)

    for i in range(ndiv):
        y_ans[i] = y_spl(x[i] / c)
    for i in range(ndiv, 2 * ndiv):
        y_ans[i] = y_spl((x[ndiv-1] / c) + (x[i - ndiv] / c))

    cam[0:ndiv] = [(y_ans[i] + y_ans[(2 * ndiv) - 1 - i]) * c / 2 for i in range(ndiv-1, -1, -1)]
    cam[0] = 0
    cam_spl = interp1d(x, cam, kind='linear', fill_value='extrapolate')
    cam_slope[0:ndiv] = np.gradient(cam_spl(x), x)
    return cam, cam_slope

def camber_thick_calc(x, coord_file):
    ndiv = len(x)
    c = x[ndiv - 1]

    cam = np.zeros(ndiv)
    cam_slope = np.zeros(ndiv)
    thick = np.zeros(ndiv)
    thick_slope = np.zeros(ndiv)

    if coord_file[:6] == "NACA00":
        m = int(coord_file[5]) / 100.0
        p = int(coord_file[6]) / 10.0
        th = int(coord_file[7:8]) / 100.0

        b1 = 0.2969
        b2 = -0.1260
        b3 = -0.3516
        b4 = 0.2843
        b5 = -0.1015

        for i in range(1, ndiv):
            thick[i] = 5 * th * (b1 * np.sqrt(x[i]) + b2 * x[i] + b3 * x[i] ** 2 + b4 * x[i] ** 3 + b5 * x[i] ** 4)
            thick_slope[i] = 5 * th * (
                    b1 / (2 * np.sqrt(x[i])) + b2 + 2 * b3 * x[i] + 3 * b4 * x[i] ** 2 + 4 * b5 * x[i] ** 3)

        thick[0] = 5 * th * (b1 * np.sqrt(x[0]) + b2 * x[0] + b3 * x[0] ** 2 + b4 * x[0] ** 3 + b5 * x[0] ** 4)
        thick_slope[0] = 2 * thick_slope[1] - thick_slope[2]

        rho = 1.1019 * th * th * c
        cam[:ndiv] = 0.0
        cam_slope[:ndiv] = 0.0

    elif coord_file[:8] == "Cylinder":
        for i in range(ndiv):
            theta = np.arccos(1. - 2 * x[i] / c)
            thick[i] = 0.5 * c * np.sin(theta)

        thick_spl = CubicSpline(x, thick)
        thick_slope[:ndiv] = thick_spl(x, 1)

        cam[:ndiv] = 0.0
        cam_slope[:ndiv] = 0.0
        rho = 0.5 * c

    elif coord_file[:9] == "FlatPlate":
        th = int(coord_file[9:13]) / 10000.0
        r = th * c / 2

        for i in range(1, ndiv - 1):
            if x[i] <= r:
                thick[i] = np.sqrt(r ** 2 - (x[i] - r) ** 2)
                thick_slope[i] = -(x[i] - r) / (np.sqrt(r ** 2 - (x[i] - r) ** 2))
            elif x[i] >= c - r:
                thick[i] = np.sqrt(r ** 2 - (x[i] - c + r) ** 2)
                thick_slope[i] = -(x[i] - c + r) / np.sqrt(r ** 2 - (x[i] - c + r) ** 2)
            else:
                thick[i] = r

        thick[0] = np.sqrt(r ** 2 - (x[0] - r) ** 2)
        thick_slope[0] = 2 * thick_slope[1] - thick_slope[2]

        thick[ndiv - 1] = np.sqrt(r ** 2 - (x[ndiv - 1] - c + r) ** 2)
        thick_slope[ndiv - 1] = 2 * thick_slope[ndiv - 2] - thick_slope[ndiv - 3]

        rho = r

        cam[:ndiv] = 0.0
        cam_slope[:ndiv] = 0.0

    else:
        coord = np.loadtxt(coord_file)

        ncoord = len(coord[:, 0])
        if 0 not in coord[:, 0]:
            raise ValueError("Airfoil file must contain leading edge coordinate (0,0)")

        nle = np.where(coord[:, 0] == 0)[0][0]
        if coord[nle, 1] != 0:
            raise ValueError("Airfoil leading edge must be at (0,0)")

        zu_spl = CubicSpline(np.flipud(coord[0:nle, 0]), np.flipud(coord[0:nle, 1]), k=1)
        zl_spl = CubicSpline(coord[nle:ncoord, 0], coord[nle:ncoord, 1], k=1)

        zu = np.zeros(ndiv)
        zl = np.zeros(ndiv)

        for i in range(ndiv):
            zu[i] = zu_spl(x[i] / c)
            zl[i] = zl_spl(x[i] / c)

        cam[:ndiv] = [(zu[i] + zl[i]) * c / 2 for i in range(ndiv)]
        thick[:ndiv] = [(zu[i] - zl[i]) * c / 2 for i in range(ndiv)]

        cam_spl = CubicSpline(x, cam)
        thick_spl = CubicSpline(x, thick)

        cam_slope[:ndiv] = cam_spl(x, 1)
        thick_slope[:ndiv] = thick_spl(x, 1)

        rho = np.loadtxt("rho")[0]

    return thick, thick_slope, rho, cam, cam_slope
