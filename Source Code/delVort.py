from abc import ABC

class delVortDef(ABC):
    pass

class delNone(delVortDef):
    pass

class delSpalart(delVortDef):
    def __init__(self, limit: int, dist: float, tol: float):
        self.limit = limit
        self.dist = dist
        self.tol = tol