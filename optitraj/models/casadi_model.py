import casadi as ca
import numpy as np 

#import abstract method from abc module
from abc import ABC, abstractmethod

class CasadiModel(ABC):
    """
    An abstract template class for defining a Casadi model.
    The user must define the states, controls, and state space.
    The user must also define the function that describes the dynamics of the system.
    Refer to the Plane class in the models module for an example.
    """    
    def __init__(self) -> None:
        self.function:ca.Function = None
        self.controls:ca.MX = None
        self.states:ca.MX = None
        self.n_states:int = None
        self.n_controls:int = None

    @abstractmethod
    def define_states(self) -> None:
        #raise NotImplementedError("Method define_states not implemented")   
        return 
        
    @abstractmethod
    def define_controls(self) -> None:
        #raise NotImplementedError("Method define_controls not implemented")
        return 
    
    @abstractmethod
    def define_state_space(self) -> None:
        #raise NotImplementedError("Method define_state_space not implemented")
        return 
    
    def rk45(self, x, u, dt, use_numeric:bool=True):
        """
        Runge-Kutta 4th order integration
        x is the current state
        u is the current control input
        dt is the time step
        use_numeric is a boolean to return the result as a numpy array
        """
        if self.function is None:
            raise RuntimeError("The function attribute is not defined.")
        
        k1 = self.function(x, u)
        k2 = self.function(x + dt/2 * k1, u)
        k3 = self.function(x + dt/2 * k2, u)
        k4 = self.function(x + dt * k3, u)
        
        next_step = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        #return as numpy row vector
        if use_numeric:
            next_step = np.array(next_step).flatten()
            return next_step
        else:
            return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    