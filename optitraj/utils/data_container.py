from typing import List, Tuple, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class MPCParams:
    Q: np.ndarray
    R: np.ndarray
    N: int
    dt: float
    
    def __post_init__(self) -> None:
        self.Q = np.array(self.Q)
        self.R = np.array(self.R)
        
    def return_as_dict(self) -> Dict:
        return {
            "Q": self.Q,
            "R": self.R,
            "N": self.N,
            "dt": self.dt
        }
        



        