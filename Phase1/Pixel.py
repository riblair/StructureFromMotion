import numpy as np
class Pixel():
    def __init__(self, RGB: tuple[(int, int, int)], u: float, v:float):
        self.u = u 
        self.v = v
        self.rgb = RGB
    
    def to_arr(self, typecast=np.float32) -> tuple[int, int]:
        return np.array([self.u, self.v], dtype=typecast)
    
    def to_hom_arr(self) -> np.ndarray:
        return np.array([[self.u],[self.v],[1]])