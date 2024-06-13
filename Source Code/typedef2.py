import numpy as np
from kinem import *

class TwoDVort:
    def __init__(self, x, z, s, vc, vx, vz):
        self.x = x
        self.z = z
        self.s = s
        self.vc = vc
        self.vx = vx
        self.vz = vz

class TwoDFlowField:
    def __init__(self, velX, velZ):
        self.velX = velX
        self.velZ = velZ
        self.u = np.array([0.])
        self.w = np.array([0.])
        self.tev = [[0 for _ in range(1)] for _ in range(6)] # The position of the elements will be in the order of 0-x,1-z,2-s,3-vc,4-vx,5-vz
        # self.tev= np.empty((0,), dtype=TwoDVort)
        self.lev = [[0 for _ in range(1)] for _ in range(6)]  # The position of the elements will be in the order of 0-x,1-z,2-s,3-vc,4-vx,5-vz
        # self.lev= np.empty((0,), dtype=TwoDVort)
        self.extv = [[0 for _ in range(1)] for _ in range(6)]  # The position of the elements will be in the order of 0-x,1-z,2-s,3-vc,4-vx,5-vz
        # self.extv= np.empty((0,), dtype=TwoDVort)