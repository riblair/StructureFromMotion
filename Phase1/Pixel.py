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
            return np.array([[self.u], [self.v]], dtype=dtype)
class Coordinate():
    def __init__(self, coord_array, norm=False):
        if norm:
            self.x = coord_array[0] / coord_array[3]
            self.y = coord_array[1] / coord_array[3]
            self.z = coord_array[2] / coord_array[3]
        else:
            self.x = coord_array[0]
            self.y = coord_array[1]
            self.z = coord_array[2]
    
    def to_arr(self, homogenous=False, dtype=np.float32) -> np.ndarray:
        if homogenous:
            return np.array([[self.x], [self.y], [self.z], [np.float32(1)]], dtype=dtype).reshape((4,1))
        else:
            return np.array([[self.x], [self.y], [self.z]], dtype=dtype)
        
    def __sub__(self, a):
        return Coordinate(np.array([self.x-a.x, self.y-a.y, self.z-a.z, 1]))
        
    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"