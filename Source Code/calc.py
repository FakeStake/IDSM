# Function for estimating a problem's time step
import numpy as np
import math 
from kinem import *
from delVort import *
from utils import *
from numba import njit 
from typedef2 import *

def update_a2a3adot(surf, dt):
    for ia in range(2, 4):
        surf.aterm[ia] = simpleTrapz(surf.downwash * np.cos(ia * surf.theta), surf.theta)
        surf.aterm[ia] = 2. * surf.aterm[ia] / (surf.uref * np.pi)
    
    surf.a0dot[0] = (surf.a0[0] - surf.a0prev[0]) / dt
    for ia in range(1,3):
        surf.adot[ia] = (surf.aterm[ia] - surf.aprev[ia]) / dt
    
    return surf

def update_atermdot(surf, dt):
    for ia in range(1, surf.naterm):
        surf.aterm[ia] = simpleTrapz(surf.downwash * np.cos(ia * surf.theta), surf.theta)
        surf.aterm[ia] = 2.0 * surf.aterm[ia] / (surf.uref * np.pi)
    
    surf.a0dot[0] = (surf.a0[0] - surf.a0prev[0]) / dt
    for ia in range(surf.naterm):
        surf.adot[ia] = (surf.aterm[ia] - surf.aprev[ia]) / dt
    
    return surf

def update_adot(surf, dt):
    surf.a0dot[0] = (surf.a0[0] - surf.a0prev[0]) / dt
    for ia in range(3):
        surf.adot[ia] = (surf.aterm[ia] - surf.aprev[ia]) / dt
    
    return surf

def update_indbound(surf, curfield):
    surf.uind[:surf.ndiv], surf.wind[:surf.ndiv] = ind_vel(np.concatenate([curfield.tev, curfield.lev, curfield.extv]), surf.bnd_x, surf.bnd_z)
    return surf

def add_indbound_b(surf, surfj):
    uind, wind = ind_vel(surfj.bv, surf.bnd_x, surf.bnd_z)
    surf.uind += uind
    surf.wind += wind
    return surf

# Function for updating the downwash
def update_downwash(surf, vels):
    for ib in range(surf.ndiv):
        surf.downwash[ib] = -(surf.kinem.u + vels[0]) * np.sin(surf.kinem.alpha) - surf.uind[ib] * np.sin(surf.kinem.alpha) + (surf.kinem.hdot - vels[1]) * np.cos(surf.kinem.alpha) - surf.wind[ib] * np.cos(surf.kinem.alpha) - surf.kinem.alphadot * (surf.x[ib] - surf.pvt * surf.c) + surf.cam_slope[ib] * (surf.uind[ib] * np.cos(surf.kinem.alpha) + (surf.kinem.u + vels[0]) * np.cos(surf.kinem.alpha) + (surf.kinem.hdot - vels[1]) * np.sin(surf.kinem.alpha) - surf.wind[ib] * np.sin(surf.kinem.alpha))
    
    return surf

# Function for a_0 and a_1 fourier coefficients
def update_a0anda1(surf):
    surf.a0[0] = simpleTrapz(surf.downwash, surf.theta)
    surf.aterm[0] = simpleTrapz(surf.downwash * np.cos(surf.theta), surf.theta)
    surf.a0[0] = -surf.a0[0] / (surf.uref * np.pi)
    surf.aterm[0] = 2.0 * surf.aterm[0] / (surf.uref * np.pi)
    return surf

# Function for calculating the fourier coefficients a_2 upwards to a_n
def update_a2toan(surf):
    for ia in range(2, surf.naterm):
        surf.aterm[ia] = simpleTrapz(surf.downwash * np.cos(ia * surf.theta), surf.theta)
        surf.aterm[ia] = 2.0 * surf.aterm[ia] / (surf.uref * np.pi)
    return surf

# Function to update the external flowfield
def update_externalvel(curfield, t):
    if isinstance(curfield.velX, CosDef):
        curfield.u[0] = curfield.velX(t)
        curfield.w[0] = curfield.velZ(t)
    elif isinstance(curfield.velX, SinDef):
        curfield.u[0] = curfield.velX(t)
        curfield.w[0] = curfield.velZ(t)
    elif isinstance(curfield.velX, ConstDef):
        curfield.u[0] = curfield.velX(t)
        curfield.w[0] = curfield.velZ(t)
    
    if isinstance(curfield.velX, StepGustDef):
        curfield.u[0] = curfield.velX(t)
    
    if isinstance(curfield.velZ, StepGustDef):
        curfield.w[0] = curfield.velZ(t)

# Function updating the dimensional kinematic parameters

def update_kinem(surf, t, dt):
    # Pitch kinematics
    if isinstance(surf.kindef.alpha, EldUpDef):
        surf.kinem.alpha = surf.kindef.alpha(t)
        surf.kinem.alphadot = np.derivative(surf.kindef.alpha, t) * surf.uref / surf.c
    elif isinstance(surf.kindef.alpha, EldUptstartDef):
        surf.kinem.alpha = surf.kindef.alpha(t)
        surf.kinem.alphadot = np.derivative(surf.kindef.alpha, t) * surf.uref / surf.c
    elif isinstance(surf.kindef.alpha, EldRampReturnDef):
        surf.kinem.alpha = surf.kindef.alpha(t)
        surf.kinem.alphadot = np.derivative(surf.kindef.alpha, t) * surf.uref / surf.c
    elif isinstance(surf.kindef.alpha, EldRampReturntstartDef):
        surf.kinem.alpha = surf.kindef.alpha(t)
        surf.kinem.alphadot= ((surf.kindef.alpha(t+dt)-surf.kindef.alpha(t))/dt)* surf.uref / surf.c
    elif isinstance(surf.kindef.alpha, ConstDef):
        surf.kinem.alpha = surf.kindef.alpha(t)
        surf.kinem.alphadot = 0.
    elif isinstance(surf.kindef.alpha, SinDef):
        surf.kinem.alpha = surf.kindef.alpha(t)
        # surf.kinem.alphadot = np.derivative(surf.kindef.alpha, t) * surf.uref / surf.c
        tt=t+dt
        surf.kinem.alphadot= ((surf.kindef.alpha(tt)-surf.kindef.alpha(t))/dt)* surf.uref / surf.c
    elif isinstance(surf.kindef.alpha, CosDef):
        surf.kinem.alpha = surf.kindef.alpha(t)
        tt=t+dt
        surf.kinem.alphadot= ((surf.kindef.alpha(tt)-surf.kindef.alpha(t))/dt)* surf.uref / surf.c
        # surf.kinem.alphadot = np.derivative(surf.kindef.alpha, t) * surf.uref / surf.c
        
    elif isinstance(surf.kindef.alpha, FileDef):
        surf.kinem.alpha = surf.kindef.alpha(t)
        tt=round(t+dt, 4)
        surf.kinem.alphadot= ((surf.kindef.alpha(tt)-surf.kindef.alpha(t))/dt)* surf.uref / surf.c
    # elif isinstance(surf.kindef.alpha, VAWTalphaDef):
    #     surf.kinem.alpha = surf.kindef.alpha(t)
    #     surf.kinem.alphadot = np.derivative(surf.kindef.alpha, t) * surf.uref / surf.c
    
    # Plunge kinematics
    if isinstance(surf.kindef.h, EldUpDef):
        surf.kinem.h = surf.kindef.h(t) * surf.c
        surf.kinem.hdot = np.derivative(surf.kindef.h, t) * surf.uref
    elif isinstance(surf.kindef.h, EldUptstartDef):
        surf.kinem.h = surf.kindef.h(t) * surf.c
        surf.kinem.hdot = np.derivative(surf.kindef.h, t) * surf.uref
    elif isinstance(surf.kindef.h, EldUpIntDef):
        surf.kinem.h = surf.kindef.h(t) * surf.c
        surf.kinem.hdot = np.derivative(surf.kindef.h, t) * surf.uref
    elif isinstance(surf.kindef.h, EldUpInttstartDef):
        surf.kinem.h = surf.kindef.h(t) * surf.c
        surf.kinem.hdot = np.derivative(surf.kindef.h, t) * surf.uref
    elif isinstance(surf.kindef.h, EldRampReturnDef):
        surf.kinem.h = surf.kindef.h(t) * surf.c
        surf.kinem.hdot = np.derivative(surf.kindef.h, t) * surf.uref
    elif isinstance(surf.kindef.h, EldRampReturntstartDef):
        surf.kinem.h = surf.kindef.h(t) * surf.c
        tt=t+dt
        surf.kinem.hdot= ((surf.kindef.h(tt)-surf.kindef.h(t))/dt)* surf.uref
        #surf.kinem.hdot = surf.kindef.h(t,True) * surf.uref
    elif isinstance(surf.kindef.h, ConstDef):
        surf.kinem.h = surf.kindef.h(t) * surf.c
        surf.kinem.hdot = 0.
    elif isinstance(surf.kindef.h, SinDef):
        surf.kinem.h = surf.kindef.h(t) * surf.c
        surf.kinem.hdot = np.derivative(surf.kindef.h, t) * surf.uref
    elif isinstance(surf.kindef.h, CosDef):
        surf.kinem.h = surf.kindef.h(t) * surf.c
        surf.kinem.hdot = np.derivative(surf.kindef.h, t) * surf.uref
    elif isinstance(surf.kindef.h, FileDef):
        surf.kinem.h = surf.kindef.h(t)
        tt=round(t+dt, 4)
        surf.kinem.hdot= ((surf.kindef.h(tt)-surf.kindef.h(t))/dt)* surf.uref
    # elif isinstance(surf.kindef.h, VAWThDef):
    #     surf.kinem.h = surf.kindef.h(t) * surf.c
    #     surf.kinem.hdot = np.derivative(surf.kindef.h, t) * surf.uref
    
    # Forward velocity
    if isinstance(surf.kindef.u, EldUpDef):
        surf.kinem.u = surf.kindef.u(t) * surf.uref
        surf.kinem.udot = np.derivative(surf.kindef.u, t) * surf.uref * surf.uref / surf.c
    elif isinstance(surf.kindef.u, EldRampReturnDef):
        surf.kinem.u = surf.kindef.u(t) * surf.uref
        surf.kinem.udot = np.derivative(surf.kindef.u, t) * surf.uref * surf.uref / surf.c
    elif isinstance(surf.kindef.u, EldRampReturntstartDef):
        surf.kinem.u = surf.kindef.u(t) * surf.uref
        surf.kinem.udot= ((surf.kindef.u(tt)-surf.kindef.u(t))/dt)* surf.uref * surf.uref/ surf.c
        #surf.kinem.udot = surf.kindef.u(t,True) * surf.uref * surf.uref / surf.c
    elif isinstance(surf.kindef.u, ConstDef):
        surf.kinem.u = surf.kindef.u(t) * surf.uref
        surf.kinem.udot = 0.
    elif isinstance(surf.kindef.u, SinDef):
        surf.kinem.u = surf.kindef.u(t) * surf.uref
        surf.kinem.udot = np.derivative(surf.kindef.u, t) * surf.uref * surf.uref / surf.c
    elif isinstance(surf.kindef.u, CosDef):
        surf.kinem.u = surf.kindef.u(t) * surf.uref
        surf.kinem.udot = np.derivative(surf.kindef.u, t) * surf.uref * surf.uref / surf.c
    elif isinstance(surf.kindef.u, LinearDef):
        surf.kinem.u = surf.kindef.u(t) * surf.uref
        surf.kinem.udot = np.derivative(surf.kindef.u, t) * surf.uref * surf.uref / surf.c
    elif isinstance(surf.kindef.u, FileDef):
        surf.kinem.u = surf.kindef.u(t)
        tt=round(t+dt, 4)
        surf.kinem.udot= ((surf.kindef.u(tt)-surf.kindef.u(t))/dt)* surf.uref * surf.uref/ surf.c
    # elif isinstance(surf.kindef.u, VAWTuDef):
    #     surf.kinem.u = surf.kindef.u(t) * surf.uref
    #     surf.kinem.udot = np.derivative(surf.kindef.u, t) * surf.uref * surf.uref / surf.c
        
    return surf


# Updates the bound vorticity distribution: eqn (2.1) in Ramesh et al. (2013)
# determines the strength of bound vortices
# determines the x, z components of the bound vortices

def update_bv(surf):
    gamma = np.zeros(surf.ndiv)
    for ib in range(1, surf.ndiv):
        gamma[ib - 1] = (surf.a0[0] * (1 + np.cos(surf.theta[ib - 1])))
        for ia in range(1, surf.naterm + 1):
            gamma[ib - 1] = gamma[ib - 1] + surf.aterm[ia - 1] * np.sin(ia * surf.theta[ib - 1]) * np.sin(surf.theta[ib - 1])
        gamma[ib - 1] = gamma[ib - 1] * surf.uref * surf.c
    
    for ib in range(1, surf.ndiv):
        surf.bv[2][ib - 1] = (gamma[ib] + gamma[ib - 1]) * (surf.theta[1] - surf.theta[0]) / 2.
        surf.bv[0][ib - 1] = (surf.bnd_x[ib] + surf.bnd_x[ib - 1]) / 2.
        surf.bv[1][ib - 1] = (surf.bnd_z[ib] + surf.bnd_z[ib - 1]) / 2.
    
    return surf

def wakeroll(surf, curfield, dt):
    vort=[]
    if curfield.tev[0] != [0]:
        ntev = len(curfield.tev[0])
    else:
        ntev=0
        
    if curfield.lev[0] != [0]:
        nlev = len(curfield.lev[0])
    else:
        nlev=0
        
    if curfield.extv[0] != [0]:    
        nextv = len(curfield.extv[0])
    else:
        nextv=0

    # Clean induced velocities
    for i in range(ntev-1):
        curfield.tev[4][i] = 0.0
        curfield.tev[5][i] = 0.0

    for i in range(nlev-1):
        curfield.lev[4][i] = 0
        curfield.lev[5][i] = 0

    for i in range(nextv-1):
        curfield.extv[4][i] = 0
        curfield.extv[5][i] = 0
        
    if curfield.tev[0] != [0]:
        vort=curfield.tev
    else:
        ntev=0
        
    if curfield.lev[0] != [0]:
        vort=[row1 + row2 for row1, row2 in zip(vort, curfield.lev)]
    else:
        nlev=0
        
    if curfield.extv[0] != [0]:    
        vort=[row1 + row2 for row1, row2 in zip(vort, curfield.extv)]
    else:
        nextv=0
    
    # Velocities induced by free vortices on each other
    vortz = mutual_ind(np.array(vort) )
    
    if ntev !=0:
        for i in range(len(curfield.tev)):
            for j in range(ntev):
                curfield.tev[i][j] = vortz[i][j]
    if nlev != 0:        
        for i in range(len(curfield.lev)):
            for j in range(ntev,ntev+nlev):
                curfield.lev[i][j-ntev] = vortz[i][j]
    if nextv != 0:        
        for i in range(len(curfield.extv)):
            for j in range(ntev+nlev,ntev+nlev+nextv):
                curfield.extv[i][j-ntev-nlev] = vortz[i][j]
                
    
    # Add the influence of velocities induced by bound vortices
    utemp = [0] * (ntev + nlev)
    wtemp = [0] * (ntev+ nlev)

    if curfield.extv[:][0] == [0]:
        utemp, wtemp = ind_velw(np.array(surf.bv), np.array(curfield.tev[:][0] + curfield.lev[:][0]), np.array(curfield.tev[:][1] + curfield.lev[:][1]))
    else:
        utemp, wtemp = ind_vel(surf.bv, curfield.tev[:][0] + curfield.lev[:][0] +  curfield.extv[:][0], curfield.tev[:][1] + curfield.lev[:][1] +  curfield.extv[:][1])

    for i in range(ntev):
        curfield.tev[4][i] += utemp[i]
        curfield.tev[5][i] += wtemp[i]

    for i in range(ntev, ntev + nlev):
        curfield.lev[4][i - ntev] += utemp[i]
        curfield.lev[5][i - ntev] += wtemp[i]

    for i in range(ntev + nlev, ntev + nlev + nextv-1):
        curfield.extv[4][i - ntev - nlev] += utemp[i]
        curfield.extv[5][i - ntev - nlev] += wtemp[i]

    # Add the influence of freestream velocities
    for i in range(ntev):
        curfield.tev[4][i] += curfield.u[0]
        curfield.tev[5][i] += curfield.w[0]

    for i in range(nlev):
        curfield.lev[4][i] += curfield.u[0]
        curfield.lev[5][i] += curfield.w[0]

    for i in range(nextv):
        curfield.extv[4][i] += curfield.u[0]
        curfield.extv[5][i] += curfield.w[0]

    # Convect free vortices with their induced velocities
    for i in range(ntev):
        curfield.tev[0][i] += dt * curfield.tev[4][i]
        curfield.tev[1][i] += dt * curfield.tev[5][i]

    for i in range(nlev):
        curfield.lev[0][i] += dt * curfield.lev[4][i]
        curfield.lev[1][i] += dt * curfield.lev[5][i]

    for i in range(nextv):
        curfield.extv[0][i] += dt * curfield.extv[4][i]
        curfield.extv[1][i] += dt * curfield.extv[5][i]

    return curfield

# Places a trailing edge vortex
def place_tev(surf, field, dt):
    ntev = len(field.tev)
    if ntev == 0:
        xloc = surf.bnd_x[surf.ndiv] + 0.5 * surf.kinem.u * dt
        zloc = surf.bnd_z[surf.ndiv]
    else:
        xloc = surf.bnd_x[surf.ndiv] + (1. / 3.) * (field.tev[ntev].x - surf.bnd_x[surf.ndiv])
        zloc = surf.bnd_z[surf.ndiv] + (1. / 3.) * (field.tev[ntev].z - surf.bnd_z[surf.ndiv])
       
    field.tev=np.append(field.tev, TwoDVort(xloc, zloc, 0., 0.02 * surf.c, 0., 0.))
    return field


def place_tev(surf, field, dt):
    nsurf = len(surf)
    ntev = len(field.tev)
    if ntev < nsurf:
        for i in range(nsurf):
            xloc = surf[i].bnd_x[-1] + 0.5 * surf[i].kinem.u * dt
            zloc = surf[i].bnd_z[-1]
            field.tev=np.append(field.tev,TwoDVort(xloc, zloc, 0., 0.02 * surf[i].c, 0., 0.))
    else:
        for i in range(nsurf):
            xloc = surf[i].bnd_x[-1] + (1. / 3.) * (field.tev[ntev - nsurf + i].x - surf[i].bnd_x[-1])
            zloc = surf[i].bnd_z[-1] + (1. / 3.) * (field.tev[ntev - nsurf + i].z - surf[i].bnd_z[-1])
            field.tev=np.append(field.tev, TwoDVort(xloc, zloc, 0., 0.02 * surf[i].c, 0., 0.))
    return field

# Places a leading edge vortex
def place_lev(surf, field, dt):
    nlev = len(field.lev)

    le_vel_x = surf.kinem.u - surf.kinem.alphadot * math.sin(surf.kinem.alpha) * surf.pvt * surf.c + surf.uind[0]
    le_vel_z = -surf.kinem.alphadot * math.cos(surf.kinem.alpha) * surf.pvt * surf.c - surf.kinem.hdot + surf.wind[0]

    if surf.levflag[0] == 0:
        xloc = surf.bnd_x[0] + 0.5 * le_vel_x * dt
        zloc = surf.bnd_z[0] + 0.5 * le_vel_z * dt
    else:
        xloc = surf.bnd_x[0] + (1. / 3.) * (field.lev[nlev].x - surf.bnd_x[0])
        zloc = surf.bnd_z[0] + (1. / 3.) * (field.lev[nlev].z - surf.bnd_z[0])

    field.lev=np.append(field.lev, TwoDVort(xloc, zloc, 0., 0.02 * surf.c, 0., 0.))

    return field


def place_lev(surf, field, dt, shed_ind):
    nsurf = len(surf)
    nlev = len(field.lev)

    for i in range(nsurf):
        if i in shed_ind:
            if surf[i].levflag[0] == 0:
                le_vel_x = surf[i].kinem.u - surf[i].kinem.alphadot * math.sin(surf[i].kinem.alpha) * surf[i].pvt * surf[i].c + surf[i].uind[0]
                le_vel_z = -surf[i].kinem.alphadot * math.cos(surf[i].kinem.alpha) * surf[i].pvt * surf[i].c - surf[i].kinem.hdot + surf[i].wind[0]
                xloc = surf[i].bnd_x[0] + 0.5 * le_vel_x * dt
                zloc = surf[i].bnd_z[0] + 0.5 * le_vel_z * dt
                field.lev=np.append(field.lev, TwoDVort(xloc, zloc, 0., 0.02 * surf[i].c, 0., 0.))
            else:
                xloc = surf[i].bnd_x[0] + (1. / 3.) * (field.lev[nlev - nsurf + i].x - surf[i].bnd_x[0])
                zloc = surf[i].bnd_z[0] + (1. / 3.) * (field.lev[nlev - nsurf + i].z - surf[i].bnd_z[0])
                field.lev=np.append(field.lev, TwoDVort(xloc, zloc, 0., 0.02 * surf[i].c, 0., 0.))
        else:
            field.lev=np.append(field.lev, TwoDVort(0., 0., 0., 0., 0., 0.))
    return field

# Function for updating the positions of the bound vortices
def update_boundpos(surf, dt):
    for i in range(surf.ndiv):
        surf.bnd_x[i] = surf.bnd_x[i] + dt * ((surf.pvt * surf.c - surf.x[i]) * math.sin(surf.kinem.alpha) * surf.kinem.alphadot - surf.kinem.u + surf.cam[i] * math.cos(surf.kinem.alpha) * surf.kinem.alphadot)
        surf.bnd_z[i] = surf.bnd_z[i] + dt * (surf.kinem.hdot + (surf.pvt * surf.c - surf.x[i]) * math.cos(surf.kinem.alpha) * surf.kinem.alphadot - surf.cam[i] * math.sin(surf.kinem.alpha) * surf.kinem.alphadot)
    return surf

# Function to calculate induced velocities by a set of vortices at a target location
def ind_vel(src, t_x, t_z):
    uind = [0.0] * len(t_x)
    wind = [0.0] * len(t_x)
    for itr in range(len(t_x)):
        for isr in range(len(src[0])):
            xdist = src[0][isr] - t_x[itr]
            zdist = src[1][isr] - t_z[itr]
            distsq = xdist * xdist + zdist * zdist
            uind[itr] = uind[itr]-src[2][isr] * zdist / (2 * math.pi * math.sqrt(src[3][isr] * src[3][isr] * src[3][isr] * src[3][isr] + distsq * distsq))
            wind[itr] = wind[itr]+src[2][isr] * xdist / (2 * math.pi * math.sqrt(src[3][isr] * src[3][isr] * src[3][isr] * src[3][isr] + distsq * distsq))
            
    return uind, wind

# Same as above except allows for only one vortex and multiple collocation points

@njit
def ind_velw(src, t_x, t_z):
    num_targets = t_x.shape[0]
    num_sources = src.shape[1]
    uind = np.zeros(num_targets)
    wind = np.zeros(num_targets)
    
    for itr in range(num_targets):
        for isr in range(num_sources):
            xdist = src[0, isr] - t_x[itr]
            zdist = src[1, isr] - t_z[itr]
            distsq = xdist * xdist + zdist * zdist
            distsq_squared = distsq * distsq
            src3_sqr = src[3, isr] * src[3, isr]
            sqrt_term = math.sqrt(src3_sqr * src3_sqr + distsq_squared)
            uind[itr] -= src[2, isr] * zdist / (2 * math.pi * sqrt_term)
            wind[itr] += src[2, isr] * xdist / (2 * math.pi * sqrt_term)
    return uind, wind

#   Used for influence coefficients
def ind_vel_obj(src, t_x, t_z):
    xdist = np.empty(())
    zdist = np.empty(())
    for i in range (0,len(src)):
        xdist = src[i].x - t_x #src[i].x
        zdist = src[i].z - t_z #src[i].z
    distsq = xdist * xdist + zdist * zdist
    uind = np.empty(())
    wind = np.empty(())
    for i in range (0,len(src)):
        uind = -src[i].s * zdist / (2 * math.pi * np.sqrt(src[i].vc * src[i].vc * src[i].vc * src[i].vc + distsq * distsq))
        wind = src[i].s * xdist / (2 * math.pi * np.sqrt(src[i].vc * src[i].vc * src[i].vc * src[i].vc + distsq * distsq))
    return uind, wind

def ind_vel_i(src, t_x, t_z):
    xdist = np.empty(())
    zdist = np.empty(())
    for i in range (0,len(src[0])):
        xdist = src[0][i] - t_x #src[i].x
        zdist = src[1][i] - t_z #src[i].z
    distsq = xdist * xdist + zdist * zdist
    uind = np.empty(())
    wind = np.empty(())
    for i in range (0,len(src[0])):
        uind = -src[2][i] * zdist / (2 * math.pi * np.sqrt(src[3][i] * src[3][i] * src[3][i] * src[3][i] + distsq * distsq))
        wind = src[2][i] * xdist / (2 * math.pi * np.sqrt(src[3][i] * src[3][i] * src[3][i] * src[3][i] + distsq * distsq))
    return uind, wind

# Function determining the effects of interacting vortices - velocities induced on each other - classical n-body problem
@njit
def mutual_ind(vorts):
    for i in range(len(vorts[0])):
        for j in range(i + 1, len(vorts[0])):
            dx = vorts[0][i] - vorts[0][j]
            dz = vorts[1][i] - vorts[1][j]
            dsq = dx * dx + dz * dz
            
            magitr = 1.0 / (2 * np.pi * np.sqrt(vorts[3][j]**4 + dsq**2))
            magjtr = 1.0 / (2 * np.pi * np.sqrt(vorts[3][i]**4 + dsq**2))

            vorts[4][j] -= dz * vorts[2][i] * magjtr
            vorts[5][j] += dx * vorts[2][i] * magjtr

            vorts[4][i] += dz * vorts[2][j] * magitr
            vorts[5][i] -= dx * vorts[2][j] * magitr
    return vorts

"""
    controlVortCount(delvort, surf, curfield)

Performs merging or deletion operation on free vortices in order to
control computational cost according to algorithm specified.

Algorithms for parameter delvort

 - delNone

    Does nothing, no vortex count control.

 - delSpalart(limit=500, dist=10, tol=1e-6)

    Merges vortices according to algorithm given in Spalart,
    P. R. (1988). Vortex methods for separated flows.

     - limit: min number of vortices present for merging to occur

     - dist: small values encourage mergin near airfoil, large values in
         wake (see paper)

     - tol: tolerance for merging, merging is less likely to occur for low
        values (see paper)

There is no universal set of parameters that work for all problem. If
using vortex control, test and calibrate choice of parameters

"""
def controlVortCount(delvort, surf_locx, surf_locz, curfield):
    if isinstance(delvort, delNone):
        pass
    elif isinstance(delvort, delSpalart):
        D0 = delvort.dist
        V0 = delvort.tol
        if len(curfield.tev) > delvort.limit:
            # Check possibility of merging the last vortex with the closest 20 vortices
            gamma_j = curfield.tev[0].s
            d_j = np.sqrt((curfield.tev[0].x - surf_locx) ** 2 + (curfield.tev[0].z - surf_locz) ** 2)
            z_j = np.sqrt(curfield.tev[0].x ** 2 + curfield.tev[0].z ** 2)

            for i in range(1, min(20, len(curfield.tev))):
                gamma_k = curfield.tev[2][i]
                d_k = np.sqrt((curfield.tev[i].x - surf_locx) ** 2 + (curfield.tev[i].z - surf_locz) ** 2)
                z_k = np.sqrt(curfield.tev[i].x ** 2 + curfield.tev[i].z ** 2)

                fact = (
                    abs(gamma_j * gamma_k) * abs(z_j - z_k)
                    / (abs(gamma_j + gamma_k) * (D0 + d_j) ** 1.5 * (D0 + d_k) ** 1.5)
                )

                if fact < V0:
                    # Merge the 2 vortices into the one at k
                    curfield.tev[i].x = (abs(gamma_j) * curfield.tev[0].x + abs(gamma_k) * curfield.tev[i].x) / (
                        abs(gamma_j + gamma_k)
                    )
                    curfield.tev[i].z = (abs(gamma_j) * curfield.tev[0].z + abs(gamma_k) * curfield.tev[i].z) / (
                        abs(gamma_j + gamma_k)
                    )
                    curfield.tev[2][i] += curfield.tev[2][0]

                    curfield.tev.pop(0)

                    break

            if len(curfield.lev) > delvort.limit:
                # Check possibility of merging the last vortex with the closest 20 vortices
                gamma_j = curfield.lev[0].s
                d_j = np.sqrt((curfield.lev[0].x - surf_locx) ** 2 + (curfield.lev[0].z - surf_locz) ** 2)
                z_j = np.sqrt(curfield.lev[0].x ** 2 + curfield.lev[0].z ** 2)

                for i in range(1, min(20, len(curfield.lev))):
                    gamma_k = curfield.lev[2][i]
                    d_k = np.sqrt((curfield.lev[i].x - surf_locx) ** 2 + (curfield.lev[i].z - surf_locz) ** 2)
                    z_k = np.sqrt(curfield.lev[i].x ** 2 + curfield.lev[i].z ** 2)

                    fact = (
                        abs(gamma_j * gamma_k) * abs(z_j - z_k)
                        / (abs(gamma_j + gamma_k) * (D0 + d_j) ** 1.5 * (D0 + d_k) ** 1.5)
                    )

                    if fact < V0:
                        # Merge the 2 vortices into the one at k
                        curfield.lev[i].x = (abs(gamma_j) * curfield.lev[0].x + abs(gamma_k) * curfield.lev[i].x) / (
                            abs(gamma_j + gamma_k)
                        )
                        curfield.lev[i].z = (abs(gamma_j) * curfield.lev[0].z + abs(gamma_k) * curfield.lev[i].z) / (
                            abs(gamma_j + gamma_k)
                        )
                        curfield.lev[2][i] += curfield.lev[2][0]

                        curfield.lev.pop(0)

                        break

def update_kinem2DOF(surf, strpar, kinem, dt, cl, cm):
    # Update previous terms
    kinem.alpha_pr = kinem.alpha
    kinem.h_pr = kinem.h

    kinem.alphadot_pr3 = kinem.alphadot_pr2
    kinem.alphadot_pr2 = kinem.alphadot_pr
    kinem.alphadot_pr = kinem.alphadot

    kinem.hdot_pr3 = kinem.hdot_pr2
    kinem.hdot_pr2 = kinem.hdot_pr
    kinem.hdot_pr = kinem.hdot

    kinem.alphaddot_pr3 = kinem.alphaddot_pr2
    kinem.alphaddot_pr2 = kinem.alphaddot_pr

    kinem.hddot_pr3 = kinem.hddot_pr2
    kinem.hddot_pr2 = kinem.hddot_pr

    # Calculate hddot and alphaddot from forces based on 2DOF structural system
    m11 = 2. / surf.c
    m12 = -strpar.x_alpha * np.cos(kinem.alpha)
    m21 = -2. * strpar.x_alpha * np.cos(kinem.alpha) / surf.c
    m22 = strpar.r_alpha * strpar.r_alpha

    R1 = (
        4 * strpar.kappa * surf.uref * surf.uref * cl / (np.pi * surf.c * surf.c)
        - 2 * strpar.w_h * strpar.w_h * (strpar.cubic_h_1 * kinem.h + strpar.cubic_h_3 * kinem.h ** 3) / surf.c
        - strpar.x_alpha * np.sin(kinem.alpha) * kinem.alphadot * kinem.alphadot
    )

    R2 = (
        8 * strpar.kappa * surf.uref * surf.uref * cm / (np.pi * surf.c * surf.c)
        - strpar.w_alpha * strpar.w_alpha * strpar.r_alpha * strpar.r_alpha
        * (strpar.cubic_alpha_1 * kinem.alpha + strpar.cubic_alpha_3 * kinem.alpha ** 3)
    )

    kinem.hddot_pr = (1 / (m11 * m22 - m21 * m12)) * (m22 * R1 - m12 * R2)
    kinem.alphaddot_pr = (1 / (m11 * m22 - m21 * m12)) * (-m21 * R1 + m11 * R2)

    # Kinematics are updated according to the 2DOF response
    kinem.alphadot = kinem.alphadot_pr + (dt / 12.) * (23 * kinem.alphaddot_pr - 16 * kinem.alphaddot_pr2 + 5 * kinem.alphaddot_pr3)
    kinem.hdot = kinem.hdot_pr + (dt / 12.) * (23 * kinem.hddot_pr - 16 * kinem.hddot_pr2 + 5 * kinem.hddot_pr3)

    kinem.alpha = kinem.alpha_pr + (dt / 12.) * (23 * kinem.alphadot_pr - 16 * kinem.alphadot_pr2 + 5 * kinem.alphadot_pr3)
    kinem.h = kinem.h_pr + (dt / 12.) * (23 * kinem.hdot_pr - 16 * kinem.hdot_pr2 + 5 * kinem.hdot_pr3)

    # Copy over these values to the TwoDSurf
    surf.kinem.alphadot = kinem.alphadot
    surf.kinem.hdot = kinem.hdot
    surf.kinem.alpha = kinem.alpha
    surf.kinem.h = kinem.h

    return surf, kinem

def panelGeo(surf):
    vor_loc = np.zeros((surf.ndiv - 1, 2))  # (x1,z1 ; x2,z2 ...) Inertial frame
    coll_loc = np.zeros((surf.ndiv - 1, 2))  # (x1,z1 ; x2,z2 ...)

    # finds geometrical panel parameters such as length, slope,
    # collocation location, and vortex location
    # length of each panel
    ds = np.sqrt((surf.x[:-1] - surf.x[1:]) ** 2 + (surf.cam[:-1] - surf.cam[1:]) ** 2)
    # Surf cam_slope does not give the correct panel locations
    cam_slope = np.arcsin((surf.cam[1:] - surf.cam[:-1]) / ds)  # [rad]

    # Normal vectors for each panel
    n = np.column_stack((-np.sin(cam_slope), np.cos(cam_slope)))
    # Tangential vectors for each panel
    tau = np.column_stack((np.cos(cam_slope), np.sin(cam_slope)))

    ## Vortex Locations
    # Located at 25% of each panel
    # Vortex locations (at 25% of panel length)
    vor_loc[:, 0] = surf.x[:-1] + 0.25 * ds * np.cos(cam_slope)
    vor_loc[:, 1] = surf.cam[:-1] + 0.25 * ds * np.sin(cam_slope)
    # Collocation points (at 75% panel chordwise)
    coll_loc[:, 0] = surf.x[:-1] + 0.75 * ds * np.cos(cam_slope)
    coll_loc[:, 1] = surf.cam[:-1] + 0.75 * ds * np.sin(cam_slope)

    return ds, cam_slope, n, tau, vor_loc, coll_loc

def IFR(surf, x, z, t):
    # Given body frame and kinematics, find global positions
    alpha = surf.kindef.alpha(t)
    X0 = -surf.kindef.u(t) * t
    Z0 = surf.kindef.h(t)
    pvt = surf.pvt

    X = (x - pvt) * np.cos(alpha) + z * np.sin(alpha) + X0 + pvt
    Z = -(x - pvt) * np.sin(alpha) + z * np.cos(alpha) + Z0

    return X, Z

def BFR(surf, X, Z, t):
    # Given inertial frame and kinematics, find body positions
    alpha = surf.kindef.alpha(t)
    X0 = -surf.kindef.u(t) * t
    Z0 = surf.kindef.h(t)
    pvt = surf.pvt

    x = (np.array(X) - X0 - pvt) * np.cos(alpha) - (np.array(Z) - Z0) * np.sin(alpha) + pvt
    z = (np.array(X) - X0 - pvt) * np.sin(alpha) + (np.array(Z) - Z0) * np.cos(alpha)

    return x, z


def refresh_vortex(surf, vor_loc):
    # Updates vortex locations to vor_loc
    for i in range(len(surf.bv[0])):
        surf.bv[0][i] = vor_loc[i, 0]
        surf.bv[1][i] = vor_loc[i, 1]
    
    return surf

def place_tev2(surf, field, dt, t):
    TE_x, TE_z = IFR(surf, surf.x[-1], surf.cam[-1], t)

    if (field.tev[0][0]) == 0:
        xloc = TE_x + 0.5 * surf.kinem.u * dt
        zloc = TE_z + 0.5 * surf.kinem.hdot * dt
        field.tev[0][0]=xloc
        field.tev[1][0]=zloc
        field.tev[3][0]=0.02 * surf.c 
    else:
        xloc = TE_x + (1. / 3.) * (field.tev[0][-1] - TE_x)
        zloc = TE_z + (1. / 3.) * (field.tev[1][-1] - TE_z)
        new=[xloc, zloc, 0., 0.02 * surf.c, 0., 0.]
        for i in range(len(new)):
            field.tev[i].append(new[i])    
    return field.tev

# Places a leading edge vortex
def place_lev2(surf, field, dt, t,i):
    LE_x, LE_z = IFR(surf, surf.x[0], surf.cam[0], t)
    le_vel_x = surf.kinem.u - surf.kinem.alphadot * np.sin(surf.kinem.alpha) * surf.pvt * surf.c + surf.uind[0]
    le_vel_z = -surf.kinem.alphadot * np.cos(surf.kinem.alpha) * surf.pvt * surf.c - surf.kinem.hdot + surf.wind[0]
    if surf.levflag[0] == 0:
        if field.lev[0][0] == 0:
            xloc = LE_x + 0.5 * le_vel_x * dt
            zloc = LE_z + 0.5 * le_vel_z * dt
            field.lev[0][0]=xloc
            field.lev[1][0]=zloc
            field.lev[3][0]=0.02 * surf.c 
        else:
            xloc = LE_x + 0.5 * le_vel_x * dt
            zloc = LE_z + 0.5 * le_vel_z * dt
            new=[xloc, zloc, 0., 0.02 * surf.c, 0., 0.]
            for i in range(len(new)):
                field.lev[i].append(new[i]) 
    else:
        xloc = LE_x + (1. / 3.) * (field.lev[0][-1] - LE_x)
        zloc = LE_z + (1. / 3.) * (field.lev[1][-1] - LE_z)
        new=[xloc, zloc, 0., 0.02 * surf.c, 0., 0.]
        for i in range(len(new)):
            field.lev[i].append(new[i])
    return field.lev

def vor_BFR(surf, t, IFR_field, curfield):
    ntev = len(IFR_field.tev[0])
    nlev = len(IFR_field.lev[0])
    vorX = IFR_field.tev[:][0] + IFR_field.lev[:][0]
    vorZ = IFR_field.tev[:][1] + IFR_field.lev[:][1]

    # Body frame conversion
    vorx=[]
    vorz=[]
    vorx, vorz = BFR(surf, vorX, vorZ, t)
    
    # Update bfr TEV
    for i in range(ntev):
        curfield.tev[0][i]= vorx[i]
        curfield.tev[1][i] = vorz[i]
    
    # Update bfr LEV
    for i in range(nlev):
        curfield.lev[0][i] = vorx[i + ntev]
        curfield.lev[1][i] = vorz[i + ntev]
    
    return curfield

def influence_coeff(surf, coll_loc, n, x_w, z_w):
    # With the surface and flowfield, determine the influence matrix "a"
    a = np.zeros((surf.ndiv, surf.ndiv))
    a[-1, :] = 1.0  # for wake portion
    vc = surf.bv[3][0]

    for i in range(surf.ndiv-1):
        for j in range(surf.ndiv):
            # i is test location, j is vortex source
            t_x = coll_loc[i, 0]
            t_z = coll_loc[i, 1]
            
            if j < surf.ndiv-1:  # Bound vortices (a_ij)
                src = [row[j] for row in surf.bv]
                xdist = src[0] - t_x
                zdist = src[1] - t_z

                distsq = xdist**2 + zdist**2  # dist_type 1
                u = -zdist / (2 * np.pi * distsq)
                w = xdist / (2 * np.pi * distsq)
                
            else:  # Wake vorticity (a_iw)
                xdist = x_w - t_x
                zdist = z_w - t_z

                distsq = xdist**2 + zdist**2  # dist type 2
                u = -zdist / (2 * np.pi * np.sqrt(vc**4 + distsq**2))
                w = xdist / (2 * np.pi * np.sqrt(vc**4 + distsq**2))
                
            a[i, j] = u * n[i, 0] + w * n[i, 1]
    return a

def mod_influence_coeff(surf, coll_loc, n, x_w, z_w, x_lev, z_lev):
    # With the surface and flowfield, determine the influence matrix "a" including the
    # modifications needed to calculate the LEV strength
    a = np.zeros((surf.ndiv, surf.ndiv))
    a[-1, :] = 1.0  # for wake portion
    vc = surf.bv[3][0]

    for i in range(surf.ndiv-1):
        for j in range(surf.ndiv):
            # i is test location, j is vortex source
            t_x = coll_loc[i, 0]
            t_z = coll_loc[i, 1]
            if j < surf.ndiv-1:  # Bound vortices (a_ij)
                src =  [row[j] for row in surf.bv]
                xdist = src[0] - t_x
                zdist = src[1] - t_z

                distsq = xdist**2 + zdist**2  # dist_type 1
                u = -zdist / (2 * np.pi * distsq)
                w = xdist / (2 * np.pi * distsq)#~u,w
            else:  # Wake vorticity (a_iw)
                xdist = x_w - t_x
                zdist = z_w - t_z

                distsq = xdist**2 + zdist**2  # dist type 2
                u = -zdist / (2 * np.pi * np.sqrt(vc**4 + distsq**2))
                w = xdist / (2 * np.pi * np.sqrt(vc**4 + distsq**2))

            a[i, j] = u * n[i, 0] + w * n[i, 1] 
            
    a1 = [row[0] for row in a]
    a[:, :-1] = a[:, 1:]  # shift over a matrix

    for i in range(surf.ndiv-1):  # calc LEV terms
        t_x = coll_loc[i, 0]
        t_z = coll_loc[i, 1]

        xdist = x_lev - t_x
        zdist = z_lev - t_z

        distsq = xdist**2 + zdist**2  # dist type 2
        u = -zdist / (2 * np.pi * np.sqrt(vc**4 + distsq**2))
        w = xdist / (2 * np.pi * np.sqrt(vc**4 + distsq**2))

        a[i, -1] = u * n[i, 0] + w * n[i, 1]

    return a, a1
