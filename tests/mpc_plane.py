import numpy as np
import casadi as ca

from casadi.casadi import MX
from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem
from optitraj.utils.data_container import MPCParams
from optitraj.models.plane import Plane
from optitraj.utils.limits import Limits, validate_limits


class PlaneOptControl(OptimalControlProblem):
    def __init__(self, 
                 mpc_params: MPCParams, 
                 casadi_model: CasadiModel,
                 state_limits:dict,
                 ctrl_limits:dict,
                 use_obs_avoidance:bool=False,
                 obs_params:dict=None) -> None:
        super().__init__(mpc_params, 
                         casadi_model,
                         state_limits,
                         ctrl_limits)
        
        self.use_obs_avoidance:bool = use_obs_avoidance
        self.obs_params:dict = obs_params
        # self.state_limits = state_limits
        # self.ctrl_limits = ctrl_limits
    
    def compute_dynamics_cost(self) -> MX:
        """
        Compute the dynamics cost for the optimal control problem
        """
        #initialize the cost
        cost = 0.0
        Q = self.mpc_params.Q
        R = self.mpc_params.R
        
        x_final = self.P[self.casadi_model.n_states:]
        
        for k in range(self.N):
            states = self.X[:, k]
            controls = self.U[:, k]
            cost += cost \
                    + (states - x_final).T @ Q @ (states - x_final) \
                    + controls.T @ R @ controls
    
        return cost
    
    def compute_total_cost(self) -> MX:
        cost = self.compute_dynamics_cost()
        return cost

        
    def solve(self, x0:np.ndarray, xF:np.ndarray, u0:np.ndarray) -> np.ndarray:
        """
        Solve the optimal control problem
        """
        state_init = ca.DM(x0)
        state_final = ca.DM(xF)
        
        X0 = ca.repmat(state_init, 1, self.N+1)
        U0 = ca.repmat(u0, 1, self.N)

        n_states = self.casadi_model.n_states
        n_controls = self.casadi_model.n_controls
        
        if self.use_obs_avoidance and self.obs_params is not None:
            #set the obstacle avoidance constraints
            pass
        else:
            num_constraints = n_states*(self.N+1)    
            lbg = ca.DM.zeros((num_constraints, 1))
            ubg  =  ca.DM.zeros((num_constraints, 1))
        
        args = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': self.pack_variables_fn(**self.lbx)['flat'],
            'ubx': self.pack_variables_fn(**self.ubx)['flat'],
        }
        
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_final   # target state
        )
        
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(self.N+1), 1),
            ca.reshape(U0, n_controls*self.N, 1)
        )
        
        # init_time = time.time()
        solution = self.solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        
        return solution
        
if __name__ == '__main__':
    plane = Plane()
    
    #let's define the limits for the states and controls
    control_limits_dict = {
        'u_phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
        'u_theta': {'min': -np.deg2rad(15), 'max': np.deg2rad(15)},
        'u_psi': {'min': -np.inf, 'max': np.inf},
        'v_cmd': {'min': 15.0, 'max': 30.0}
    }
    state_limits_dict = {
        'x': {'min': -np.inf, 'max': np.inf},
        'y': {'min': -np.inf, 'max': np.inf},
        'z': {'min': 15, 'max': 100},
        'phi': {'min': -np.deg2rad(45), 'max': np.deg2rad(45)},
        'theta': {'min': -np.deg2rad(15), 'max': np.deg2rad(15)},
        'psi': {'min': -np.pi, 'max': np.pi},
        'v': {'min': 15, 'max': 30.0}
    }

    # # Call the validation function
    # validate_limits(control_limits_dict)
    
    #create a diagonal matrix for Q and R based on the number of states and controls
    Q = np.array([1, 1, 1, 0, 0, 0, 0])
    # Q = np.eye(plane.n_states)
    Q = np.diag(Q)
    print("Q: ", Q)
    R = np.eye(plane.n_controls)
    
    
    params = MPCParams(Q=Q, R=R, N=10, dt=0.1)
    mpc = PlaneOptControl(mpc_params=params, 
                          casadi_model=plane,
                          state_limits=state_limits_dict,
                          ctrl_limits=control_limits_dict)
    mpc.init_optimization()
    
    x0 = np.array([0, 0, 50, 0, 0, 0, 15])
    xF = np.array([100, 100, 60, 0, 0, 0, 30])
    u0 = np.array([0, 0, 0, 15])
    
    #solve the optimal control problem

    N = 100
    idx = 1
    
    for i in range(N):
        sol = mpc.solve_and_get_solution(x0, xF, u0)
        states = sol['states']
        ctrl = sol['controls']   
        next_state = []
        next_ctrl = []
        
        for i, state_name in enumerate(state_limits_dict.keys()):
            next_state.append(states[state_name][idx])
        
        for i, ctrl_name in enumerate(control_limits_dict.keys()):
            next_ctrl.append(ctrl[ctrl_name][idx])
        x0 = np.array(next_state)
        u0 = np.array(next_ctrl)
        
        
        distance = np.linalg.norm(np.array(next_state[0:2]) - xF[0:2])
        print('Distance to target: ', distance)
        print("position: ", x0[0:3])
        if distance < 5.0:
            print('Optimization complete')
        
print('Optimization complete')
print("States: ", next_state)