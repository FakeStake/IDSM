from math import *
import numpy as np
from numba import njit

class MotionDef:
    pass

class KinemPar:
    def __init__(self, alpha, h, alphadot, hdot, u, udot):
        self.alpha = alpha
        self.h = h
        self.alphadot = alphadot
        self.hdot = hdot
        self.u = u
        self.udot = udot

class KinemDef:
    def __init__(self, alpha, h, u):
        self.alpha = alpha
        self.h = h
        self.u = u

class EldUpDef(MotionDef):
    def __init__(self, amp, K, a):
        self.amp = amp
        self.K = K
        self.a = a

    def __call__(self, t):
        sm = pi*pi*self.K / (2*self.amp*(1 - self.a))
        t1 = 1.
        t2 = t1 + (self.amp / (2*self.K))
        return ((self.K/sm)*log(cosh(sm*(t - t1))/cosh(sm*(t - t2)))) + (self.amp / 2)

class EldUptstartDef(MotionDef):
    def __init__(self, amp, K, a, tstart):
        self.amp = amp
        self.K = K
        self.a = a
        self.tstart = tstart

    def __call__(self, t):
        sm = pi*pi*self.K / (2*self.amp*(1 - self.a))
        t1 = self.tstart
        t2 = t1 + (self.amp / (2*self.K))
        return ((self.K/sm)*log(cosh(sm*(t - t1))/cosh(sm*(t - t2)))) + (self.amp / 2)

class EldRampReturnDef(MotionDef):
    def __init__(self, amp, K, a):
        self.amp = amp
        self.K = K
        self.a = a

    def __call__(self, tt):
        fr = self.K / (pi*abs(self.amp))
        t1 = 1.
        t2 = t1 + (1. /(2*pi*fr))
        t3 = t2 + ((1/(4*fr)) - (1/(2*pi*fr)))
        t4 = t3 + (1. /(2*pi*fr))
        t5 = t4 + 1.

        nstep = int(round(t5 / 0.015)) + 1
        g = np.zeros(nstep)
        t = np.zeros(nstep)

        for i in range(nstep):
            t[i] = (i-1.) * 0.015
            g[i] = log((cosh(self.a*(t[i] - t1))*cosh(self.a*(t[i] - t4)))) / (cosh(self.a*(t[i] - t2))*cosh(self.a*(t[i] - t3)))
        maxg = np.max(g)

        gg = log((cosh(self.a*(tt - t1))*cosh(self.a*(tt - t4)))) / (cosh(self.a*(tt - t2))*cosh(self.a*(tt - t3)))

        return self.amp * gg / maxg

@njit
def compute(amp, K, a, tstart, dt, tt):
    fr = K / (pi * abs(amp))
    t1 = tstart
    t2 = t1 + (1. / (2 * pi * fr))
    t3 = t2 + ((1 / (4 * fr)) - (1 / (2 * pi * fr)))
    t4 = t3 + (1. / (2 * pi * fr))
    t5 = t4 + t1

    nstep = int(round(t5 / dt)) + 1
    g = np.zeros(nstep)
    t = np.zeros(nstep)
    
    for i in range(nstep):
        t[i] = (i - 1.) * dt
        g[i] = log((cosh(a * (t[i] - t1)) * cosh(a * (t[i] - t4))) / 
                   (cosh(a * (t[i] - t2)) * cosh(a * (t[i] - t3))))
    
    maxg = np.max(g)
    
    gg = log((cosh(a * (tt - t1)) * cosh(a * (tt - t4))) / 
             (cosh(a * (tt - t2)) * cosh(a * (tt - t3))))
    
    return amp * gg / maxg

class EldRampReturntstartDef:
    def __init__(self, amp, K, a, tstart, dt):
        self.amp = amp
        self.K = K
        self.a = a
        self.tstart = tstart
        self.dt = dt

    def __call__(self, tt):
        return compute(self.amp, self.K, self.a, self.tstart, self.dt, tt)


class ConstDef(MotionDef):
    def __init__(self, amp):
        self.amp = amp

    def __call__(self, t):
        return self.amp

class LinearDef(MotionDef):
    def __init__(self, tstart, vstart, vend, length):
        self.tstart = tstart
        self.vstart = vstart
        self.vend = vend
        self.length = length

    def __call__(self, t):
        if t < self.tstart:
            return self.vstart
        elif t > self.tstart + self.length:
            return self.vend
        else:
            return self.vstart + (self.vend - self.vstart) / self.length * (t - self.tstart)

class BendingDef(MotionDef):
    def __init__(self, spl, scale, k, phi):
        self.spl = spl
        self.scale = scale
        self.k = k
        self.phi = phi

class SinDef(MotionDef):
    def __init__(self, mean, amp, k, phi):
        self.mean = mean
        self.amp = amp
        self.k = k
        self.phi = phi

    def __call__(self, t):
        return self.mean + self.amp * sin(2*self.k*t + self.phi)

class CosDef(MotionDef):
    def __init__(self, mean, amp, k, phi):
        self.mean = mean
        self.amp = amp
        self.k = k
        self.phi = phi

    def __call__(self, t):
        return self.mean + self.amp * cos(2*self.k*t + self.phi)

class StepGustDef(MotionDef):
    def __init__(self, amp, tstart, tgust):
        self.amp = amp
        self.tstart = tstart
        self.tgust = tgust

    def __call__(self, t):
        if t >= self.tstart and t <= self.tstart + self.tgust:
            return self.amp
        else:
            return 0.

class EldUpIntDef(MotionDef):
    def __init__(self, amp, K, a, tdur):
        self.amp = amp
        self.K = K
        self.a = a
        self.tdur = tdur

    def __call__(self, tt):
        t1 = 1.
        t2 = t1 + self.tdur

        nstep = int(round(t2 / 0.015)) + 1
        g = np.zeros(nstep)
        t = np.zeros(nstep)

        for i in range(nstep):
            t[i] = (i-1.) * 0.015
            g[i] = log(cosh(self.a*(t[i] - t1))) / cosh(self.a*(t[i] - t2))
        maxg = np.max(g)

        gg = log(cosh(self.a*(tt - t1))) / cosh(self.a*(tt - t2))

        return self.amp * gg / maxg

class EldUpInttstartDef(MotionDef):
    def __init__(self, amp, K, a, tdur, tstart):
        self.amp = amp
        self.K = K
        self.a = a
        self.tdur = tdur
        self.tstart = tstart

    def __call__(self, tt):
        t1 = self.tstart
        t2 = t1 + self.tdur

        nstep = int(round(t2 / 0.015)) + 1
        g = np.zeros(nstep)
        t = np.zeros(nstep)

        for i in range(nstep):
            t[i] = (i-1.) * 0.015
            g[i] = log(cosh(self.a*(t[i] - t1))) / cosh(self.a*(t[i] - t2))
        maxg = np.max(g)

        gg = log(cosh(self.a*(tt - t1))) / cosh(self.a*(tt - t2))

        return self.amp * gg / maxg

class FileDef:
    def __init__(self, data):
        self.t=data[:,0]
        self.alpha = data[:, 1]
        # self.href = data[:,2]
        # self.uref = data[:,3]

    def __call__(self, tt):
        tot=self.t
        a=np.where(tot == tt)[0]
        a=self.alpha[a]
        if isinstance(a, np.ndarray):
            a=a[0]
        return np.radians(a)
    
class hDef:
    def __init__(self, data):
        self.t=data[:,0]
        self.href = data[:,2]

    def __call__(self, tt):
        tot=self.t
        a=np.where(tot == tt)[0]
        a=self.href[a]
        if isinstance(a, np.ndarray):
            a=a[0]
        return a
    
class uDef:
    def __init__(self, data):
        self.t=data[:,0]
        self.uref = data[:,3]

    def __call__(self, tt):
        tot=self.t
        a=np.where(tot == tt)[0]
        a=self.uref[a]
        if isinstance(a, np.ndarray):
            a=a[0]
        return a