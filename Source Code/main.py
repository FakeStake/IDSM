import os
import numpy as np
import matplotlib.pyplot as plt
from kinem import *
from typedefs import *
from LVE import *

def main(geo,file,app,motion,ndiv=101,nsteps=500,rho=1.225,pvt=0.25,LESP=0.2,k_value=0.1,amp=90,delay=10,phi=0,mean=0,save_file_force_data=False,file_name_file_force_data='',save_flow_force_data=False,tev_file_name='', ani_count=50):

    # Kinematics
    data=0
    mean=0
    ndiv=float(ndiv)
    rho=float(rho)
    pvt=float(pvt)
    LESP=float(LESP)


    if motion == "User defined":
        data = np.loadtxt(file)
        t=data[:,0]
        dtstar = t[1]-t[0]
        tstart=t[0]
        
        alphadef=FileDef(data)
        hdef = hDef(data)#
        udef = uDef(data)#
        nsteps = nsteps-2
    elif motion == "Pitch ramp motion":
        amp = float(amp)
        k = float(k_value)
        tstart = float(delay)
        a = (np.pi**2 * k * 180) / (2 * amp * np.pi * (1 - 0.1))
        dtstar=0.015

        alphadef = EldRampReturntstartDef(amp * np.pi / 180, k, a, tstart, dtstar)

        hdef = ConstDef(0)#
        udef = ConstDef(1)#
    elif motion == "Sine motion":
        amp = float(amp)
        k = float(k_value)
        phi=float(phi)
        dtstar=0.015
        
        alphadef = SinDef(mean, amp * np.pi / 180, k, phi)

        hdef = ConstDef(0)#
        udef = ConstDef(1)#
    elif motion == "Cosine motion":
        amp = float(amp)
        k = float(k_value)
        phi=float(phi)
        dtstar=0.015
        
        alphadef = CosDef(mean, amp * np.pi / 180, k, phi)
        hdef = ConstDef(0)#
        udef = ConstDef(1)#


    full_kinem = KinemDef(alphadef, hdef, udef)
   
    # Geometry
    lespcrit = np.empty((1))
    lespcrit[0]=LESP

    velX=[]
    velZ=[]
    surf = TwoDSurf(geo, dtstar, pvt, full_kinem, lespcrit, ndiv=101, camberType="linear", rho=1.225) # ndiv ~ number panel division
    curfield = TwoDFlowField(velX,velZ) # vortex structure

    print("Running LVE")
    ttime=LVE(surf, curfield, app, nsteps, dtstar, ani_count,save_file_force_data,file_name_file_force_data,save_flow_force_data,tev_file_name)
    print("Ending LVE")
    print("end simluation")

    return ttime
