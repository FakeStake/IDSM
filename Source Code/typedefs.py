"""
    TwoDVort(x,z,s,vc,vx,vz)

Defines a 2D vortex at `(x,z)` with vorticity `s` and vortex-core
radius `vc`.

`vx` and `vz` are induced velocity at centroid used in vortex
interaction calculations

"""

import numpy as np
from kinem import *
from utils import *
from calc import *

# class TwoDVort:
#     def __init__(self, x, z, s, vc, vx, vz):
#         self.x = x
#         self.z = z
#         self.s = s
#         self.vc = vc
#         self.vx = vx
#         self.vz = vz

# class TwoDFlowField:
#     def __init__(self, velX=ConstDef(0), velZ=ConstDef(0)):
#         self.velX = velX
#         self.velZ = velZ
#         self.u = np.array([0.])
#         self.w = np.array([0.])
#         self.tev = np.array([0.])
#         self.lev = np.array([0.])
#         self.extv = np.array([0.])

class TwoDSurf:
    def __init__(self, coord_file, dt, pvt, kindef, lespcrit=None, c=1.0, uref=1.0, ndiv=70, naterm=35, initpos=[0.0, 0.0], rho=0.04, camberType="radial"):
        self.c = c
        self.uref = uref
        self.coord_file = coord_file
        self.pvt = pvt
        self.ndiv = ndiv
        self.naterm = naterm
        self.kindef = kindef
        self.cam = np.zeros(ndiv)
        self.cam_slope = np.zeros(ndiv)
        self.theta = np.zeros(ndiv)
        self.x = np.zeros(ndiv)
        self.kinem = KinemPar(0, 0, 0, 0, 0, 0)
        self.bnd_x = np.zeros(ndiv)
        self.bnd_z = np.zeros(ndiv)
        self.uind = np.zeros(ndiv)
        self.wind = np.zeros(ndiv)
        self.downwash = np.zeros(ndiv)
        self.a0 = np.zeros(1)
        self.aterm = np.zeros(naterm)
        self.a0dot = np.zeros(1)
        self.adot = np.zeros(naterm)
        self.a0prev = np.zeros(1)
        self.aprev = np.zeros(naterm)
        self.bv = []
        self.lespcrit = lespcrit
        self.levflag = [0]
        self.initpos = initpos
        self.rho = rho

        if camberType == "radial":
            dtheta = np.pi / (ndiv - 1)
            for ib in range(ndiv):
                self.theta[ib] = (ib - 1) * dtheta
                self.x[ib] = c / 2. * (1 - np.cos(self.theta[ib]))
        elif camberType == "linear":
            self.x = np.linspace(0, self.c, self.ndiv)

        if coord_file != "FlatPlate":
            self.cam, self.cam_slope = camber_calc(self.x, coord_file)

        if isinstance(kindef.alpha, EldUpDef):
            self.kinem.alpha = kindef.alpha(0)
            self.kinem.alphadot = np.derivative(kindef.alpha, 0) * uref / c
        elif isinstance(kindef.alpha, EldRampReturnDef):
            self.kinem.alpha = kindef.alpha(0)
            self.kinem.alphadot = np.derivative(kindef.alpha, 0) * uref / c
        elif isinstance(kindef.alpha, ConstDef):
            self.kinem.alpha = kindef.alpha(0)
            self.kinem.alphadot = 0
        elif isinstance(kindef.alpha, SinDef):
            self.kinem.alpha = kindef.alpha(0)
            self.kinem.alphadot = (kindef.alpha(dt)- kindef.alpha(0))/dt * uref / c
        elif isinstance(kindef.alpha, CosDef):
            self.kinem.alpha = kindef.alpha(0)
            self.kinem.alphadot = (kindef.alpha(dt)- kindef.alpha(0))/dt * uref / c
        elif isinstance(kindef.alpha, FileDef):
            self.kinem.alpha = kindef.alpha(0)
            self.kinem.alphadot = (kindef.alpha(dt)- kindef.alpha(0))/dt * uref / c

        if isinstance(kindef.h, EldUpDef):
            self.kinem.h = kindef.h(0) * c
            self.kinem.hdot = np.derivative(kindef.h, 0) * uref
        elif isinstance(kindef.h, EldUpIntDef):
            self.kinem.h = kindef.h(0) * c
            self.kinem.hdot = np.derivative(kindef.h, 0) * uref
        elif isinstance(kindef.h, EldRampReturnDef):
            self.kinem.h = kindef.h(0) * c
            self.kinem.hdot = np.derivative(kindef.h, 0) * uref
        elif isinstance(kindef.h, ConstDef):
            self.kinem.h = kindef.h(0) * c
            self.kinem.hdot = 0
        elif isinstance(kindef.h, SinDef):
            self.kinem.h = kindef.h(0) * c
            self.kinem.hdot = np.derivative(kindef.h, 0) * uref
        elif isinstance(kindef.h, CosDef):
            self.kinem.h = kindef.h(0) * c
            self.kinem.hdot = np.derivative(kindef.h, 0) * uref
        elif isinstance(kindef.h, FileDef):
            self.kinem.h = kindef.h(0)
            self.kinem.hdot = (kindef.h(dt)- kindef.h(0))/dt * uref

        if isinstance(kindef.u, EldUpDef):
            self.kinem.u = kindef.u(0) * uref
            self.kinem.udot = np.derivative(kindef.u, 0) * uref * uref / c
        elif isinstance(kindef.u, EldRampReturnDef):
            self.kinem.u, self.kinem.udot = kindef.u(0)
            self.kinem.u = self.kinem.u * uref
            self.kinem.udot = self.kinem.udot * uref * uref / c
        elif isinstance(kindef.u, ConstDef):
            self.kinem.u = kindef.u(0) * uref
            self.kinem.udot = 0
        elif isinstance(kindef.alpha, FileDef):
            self.kinem.u = kindef.u(0) * uref
            self.kinem.udot = (kindef.u(dt)- kindef.u(0))/dt * uref * uref / c

        for i in range(ndiv):
            self.bnd_x[i] = -((c - pvt * c) + ((pvt * c - self.x[i]) * np.cos(self.kinem.alpha))) + (self.cam[i] * np.sin(self.kinem.alpha)) + initpos[0]
            self.bnd_z[i] = self.kinem.h + ((pvt * c - self.x[i]) * np.sin(self.kinem.alpha)) + (self.cam[i] * np.cos(self.kinem.alpha)) + initpos[1]

        self.uind = np.zeros(ndiv)
        self.wind = np.zeros(ndiv)
        self.downwash = np.zeros(ndiv)
        self.a0 = np.zeros(1)
        self.a0dot = np.zeros(1)
        self.aterm = np.zeros(naterm)
        self.adot = np.zeros(naterm)
        self.a0prev = np.zeros(1)
        self.aprev = np.zeros(naterm)
        self.bv = [[0 for _ in range(ndiv-1)] for _ in range(6)]
        for i in range(ndiv - 1):
            self.bv[3][i]=.02*c
        self.levflag = [0]

class TwoDOFPar:
    def __init__(self, x_alpha, r_alpha, kappa, w_alpha, w_h, w_alphadot, w_hdot, cubic_h_1, cubic_h_3, cubic_alpha_1, cubic_alpha_3):
        self.x_alpha = x_alpha
        self.r_alpha = r_alpha
        self.kappa = kappa
        self.w_alpha = w_alpha
        self.w_h = w_h
        self.w_alphadot = w_alphadot
        self.w_hdot = w_hdot
        self.cubic_h_1 = cubic_h_1
        self.cubic_h_3 = cubic_h_3
        self.cubic_alpha_1 = cubic_alpha_1
        self.cubic_alpha_3 = cubic_alpha_3

#For airfoil with 2DOF in pitch and plunge
class KinemPar2DOF:
    def __init__(self, alpha, h, alphadot, hdot, u):
        self.alpha = alpha
        self.h = h
        self.alphadot = alphadot
        self.hdot = hdot
        self.u = u
        self.udot = 0.0
        self.alphaddot = 0.0
        self.hddot = 0.0
        self.alpha_pr = alpha
        self.h_pr = h
        self.alphadot_pr = alphadot
        self.alphadot_pr2 = alphadot
        self.alphadot_pr3 = alphadot
        self.hdot_pr = hdot
        self.hdot_pr2 = hdot
        self.hdot_pr3 = hdot
        self.alphaddot_pr = 0.0
        self.alphaddot_pr2 = 0.0
        self.alphaddot_pr3 = 0.0
        self.hddot_pr = 0.0
        self.hddot_pr2 = 0.0
        self.hddot_pr3 = 0.0

class KelvinCondition:
    def __init__(self, surf, field):
        self.surf = surf
        self.field = field

def kelvin_condition(kelv, tev_iter):
    nlev = len(kelv.field.lev)
    ntev = len(kelv.field.tev)

    uprev, wprev = ind_vel([kelv.field.tev[ntev]], kelv.surf.bnd_x, kelv.surf.bnd_z)

    # Update the TEV strength
    kelv.field.tev[ntev].s = tev_iter[0]

    unow, wnow = ind_vel([kelv.field.tev[ntev]], kelv.surf.bnd_x, kelv.surf.bnd_z)

    kelv.surf.uind -= uprev - unow
    kelv.surf.wind -= wprev - wnow

    # Calculate downwash
    update_downwash(kelv.surf, [kelv.field.u[0], kelv.field.w[0]])

    # Calculate first two Fourier coefficients
    update_a0anda1(kelv.surf)

    val = kelv.surf.uref * kelv.surf.c * pi * (kelv.surf.a0[0] + kelv.surf.aterm[0] / 2.0) - \
          kelv.surf.uref * kelv.surf.c * pi * (kelv.surf.a0prev[0] + kelv.surf.aprev[0] / 2.0) + \
          kelv.field.tev[ntev].s

    return val


class KelvinConditionMult:
    def __init__(self, surf, field):
        self.surf = surf
        self.field = field

    def __call__(self, tev_iter):
        nsurf = len(self.surf)
        val = np.zeros(nsurf)
        nlev = len(self.field.lev)
        ntev = len(self.field.tev)
        tev_list = self.field.tev[ntev-nsurf:ntev]
        
        for i in range(nsurf):
            bv_list = []
            for j in range(nsurf):
                if i != j:
                    bv_list.extend(self.surf[j].bv)
            
            uprev, wprev = ind_vel(tev_list + bv_list, self.surf[i].bnd_x, self.surf[i].bnd_z)

            # Update the TEV strength
            self.field.tev[ntev-nsurf+i].s = tev_iter[i]

            unow, wnow = ind_vel(tev_list + bv_list, self.surf[i].bnd_x, self.surf[i].bnd_z)

            self.surf[i].uind -= uprev - unow
            self.surf[i].wind -= wprev - wnow

            # Calculate downwash
            update_downwash(self.surf[i], [self.field.u[0], self.field.w[0]])
        
        for i in range(nsurf):
            # Update Fourier coefficients and bv strength
            update_a0anda1(self.surf[i])
            update_a2toan(self.surf[i])
            update_bv(self.surf[i])
            
            val[i] = self.surf[i].uref * self.surf[i].c * np.pi * (self.surf[i].a0[0] + self.surf[i].aterm[0] / 2.0) - \
                     self.surf[i].uref * self.surf[i].c * np.pi * (self.surf[i].a0prev[0] + self.surf[i].aprev[0] / 2.0) + \
                     self.field.tev[ntev-nsurf+i].s
        
        return val


class KelvinKutta:
    def __init__(self, surf, field):
        self.surf = surf
        self.field = field

    def __call__(self, v_iter):
        val = np.zeros(2)
        nlev = len(self.field.lev)
        ntev = len(self.field.tev)
        
        uprev, wprev = ind_vel([self.field.tev[ntev], self.field.lev[nlev]], self.surf.bnd_x, self.surf.bnd_z)

        # Update the TEV and LEV strengths
        self.field.tev[ntev].s = v_iter[0]
        self.field.lev[nlev].s = v_iter[1]

        unow, wnow = ind_vel([self.field.tev[ntev], self.field.lev[nlev]], self.surf.bnd_x, self.surf.bnd_z)

        self.surf.uind -= uprev - unow
        self.surf.wind -= wprev - wnow

        # Calculate downwash
        update_downwash(self.surf, [self.field.u[0], self.field.w[0]])

        # Calculate first two Fourier coefficients
        update_a0anda1(self.surf)

        val[0] = self.surf.uref * self.surf.c * np.pi * (self.surf.a0[0] + self.surf.aterm[0] / 2.0) - \
                 self.surf.uref * self.surf.c * np.pi * (self.surf.a0prev[0] + self.surf.aprev[0] / 2.0) + \
                 self.field.tev[ntev].s + self.field.lev[nlev].s

        if self.surf.a0[0] > 0:
            lesp_cond = self.surf.lespcrit[0]
        else:
            lesp_cond = -self.surf.lespcrit[0]
        
        val[1] = self.surf.a0[0] - lesp_cond

        return val


class KelvinKuttaMult:
    def __init__(self, surf, field, shed_ind):
        self.surf = surf
        self.field = field
        self.shed_ind = shed_ind

    def __call__(self, v_iter):
        nsurf = len(self.surf)
        nshed = len(self.shed_ind)
        val = np.zeros(nsurf + nshed)

        nlev = len(self.field.lev)
        ntev = len(self.field.tev)

        tev_list = self.field.tev[ntev - nsurf: ntev]
        lev_list = [self.field.lev[nlev - nsurf + index] for index in self.shed_ind]

        levcount = 0
        for i in range(nsurf):
            bv_list = [self.surf[j].bv for j in range(nsurf) if i != j]
            uprev, wprev = ind_vel(tev_list + lev_list + bv_list, self.surf[i].bnd_x, self.surf[i].bnd_z)

            # Update the shed vortex strengths
            self.field.tev[ntev - nsurf + i].s = v_iter[i]

            if i in self.shed_ind:
                levcount += 1
                self.field.lev[nlev - nsurf + i].s = v_iter[nsurf + levcount]

            unow, wnow = ind_vel(tev_list + lev_list + bv_list, self.surf[i].bnd_x, self.surf[i].bnd_z)

            self.surf[i].uind -= uprev - unow
            self.surf[i].wind -= wprev - wnow

            # Calculate downwash
            update_downwash(self.surf[i], [self.field.u[0], self.field.w[0]])

        levcount = 0
        for i in range(nsurf):
            # Update Fourier coefficients and bv strength
            update_a0anda1(self.surf[i])
            update_a2toan(self.surf[i])
            update_bv(self.surf[i])

            if i in self.shed_ind:
                val[i] = self.surf[i].uref * self.surf[i].c * np.pi * (self.surf[i].a0[0] + self.surf[i].aterm[0] / 2.0) - \
                         self.surf[i].uref * self.surf[i].c * np.pi * (self.surf[i].a0prev[0] + self.surf[i].aprev[0] / 2.0) + \
                         self.field.tev[ntev - nsurf + i].s + self.field.lev[nlev - nsurf + i].s
                levcount += 1
                if self.surf[i].a0[0] > 0:
                    lesp_cond = self.surf[i].lespcrit[0]
                else:
                    lesp_cond = -self.surf[i].lespcrit[0]

                val[levcount + nsurf] = self.surf[i].a0[0] - lesp_cond
            else:
                val[i] = self.surf[i].uref * self.surf[i].c * np.pi * (self.surf[i].a0[0] + self.surf[i].aterm[0] / 2.0) - \
                         self.surf[i].uref * self.surf[i].c * np.pi * (self.surf[i].a0prev[0] + self.surf[i].aprev[0] / 2.0) + \
                         self.field.tev[ntev - nsurf + i].s

        return val


class MeshGrid:
    def __init__(self, t, alpha, tev, lev, camX, camZ, x, z, uMat, wMat, velMag):
        self.t = t
        self.alpha = alpha
        self.tev = tev
        self.lev = lev
        self.camX = camX
        self.camZ = camZ
        self.x = x
        self.z = z
        self.uMat = uMat
        self.wMat = wMat
        self.velMag = velMag

    # @staticmethod
    def Meshgrid(surf, tevs, levs, offset, t, width=100, view="square"):
        farBnd = surf.x[-1] + surf.c * offset
        nearBnd = surf.x[0] - surf.c * offset
        zBnd = (farBnd - nearBnd) / 2

        if view == "wake":
            farBnd = 2 * farBnd
        elif view == "largewake":
            farBnd = 3 * farBnd
            zBnd = 2 * zBnd
        elif view == "longwake":
            farBnd = 5 * farBnd
            zBnd = 2 * zBnd
        elif view == "UI Window":
            farBnd = 2 * farBnd
            zBnd = 3/4 * farBnd

        X0 = -surf.kindef.u(t) * t
        Z0 = surf.kindef.h(t)

        lowX = nearBnd + X0 - surf.pvt * surf.c
        uppX = farBnd + X0 - surf.pvt * surf.c
        lowZ = -zBnd + Z0
        uppZ = zBnd + Z0

        camX, camZ = IFR(surf, surf.x, surf.cam, t)

        step = (farBnd - nearBnd) / (width - 1)
        rangeX = np.arange(lowX, uppX + step, step)
        height = int(zBnd * 2 / step) + 1
        x = np.tile(rangeX, (height, 1))

        rangeZ = np.arange(uppZ, lowZ - step, -step)
        z = np.tile(rangeZ, (x.shape[1], 1)).T

        uMat = np.zeros_like(x) + surf.kinem.u
        wMat = np.zeros_like(x) + surf.kinem.hdot
        velMag = np.zeros_like(x)

        t = 0
        alpha = surf.kinem.alpha
        circ = np.zeros(surf.ndiv - 1)

        tevX = []
        tevZ = []
        for tev in tevs:
            vx = tev.x
            vz = tev.z
            if lowX <= vx <= uppX and lowZ <= vz <= uppZ:
                tevX.append(vx)
                tevZ.append(vz)
        tev = np.column_stack((tevX, tevZ))

        levX = []
        levZ = []
        if len(levs) > 0:
            for lev in levs:
                vx = lev.x
                vz = lev.z
                if lowX <= vx <= uppX and lowZ <= vz <= uppZ:
                    levX.append(vx)
                    levZ.append(vz)
        lev = np.column_stack((levX, levZ))

        return MeshGrid(t, alpha, tev, lev, camX, camZ, x, z, uMat, wMat, velMag)
