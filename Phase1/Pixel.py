import numpy as np
class Pixel():
    def __init__(self, RGB: tuple[(int, int, int)], u: float, v:float):
        self.u = u 
        self.v = v
        self.rgb = RGB
    
    def to_arr(self, homogenous=False, dtype=np.float32) -> np.ndarray:
        if homogenous:
            return np.array([[self.u],[self.v],[1]], dtype=dtype)
        else:
            return np.array([self.u, self.v], dtype=dtype)
class Coordinate():
    def __init__(self, coord_array):
        self.x = coord_array[0]
        self.y = coord_array[1]
        self.z = coord_array[2]
    
    def to_arr(self, homogenous=False, dtype=np.float32) -> np.ndarray:
        if homogenous:
            return np.array([[self.x], [self.y], [self.z], [np.float32(1)]], dtype=dtype)
        else:
            return np.array([[self.x], [self.y], [self.z]], dtype=dtype)
        