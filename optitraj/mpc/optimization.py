import casadi as ca
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from optitraj.utils.data_container import MPCParams
from optitraj.models.casadi_model import CasadiModel
from optitraj.utils.limits import Limits, validate_limits


class OptimalControlProblem(ABC):
    """
    Represents an optimal control problem for model predictive control
    Refer to 
    """

    def __init__(self,
                 mpc_params: MPCParams,
                 casadi_model: CasadiModel) -> None:

        self.nlp: Dict = None
        self.solver = None
        self.is_initialized: bool = False
        self.cost: float = 0.0

        self.mpc_params: MPCParams = mpc_params
        self.N: np.ndarray[float] = mpc_params.N
        self.dt: np.ndarray[float] = mpc_params.dt
        self.Q: np.ndarray[float] = mpc_params.Q
        self.R: np.ndarray[float] = mpc_params.R
        self.casadi_model: CasadiModel = casadi_model
        self.state_limits: dict = casadi_model.state_limits
        self.ctrl_limits: dict = casadi_model.control_limits

        if self.state_limits is None:
            raise ValueError("State limits not defined.")
        if self.ctrl_limits is None:
            raise ValueError("Control limits not defined.")

        validate_limits(self.state_limits, limit_type="state")
        validate_limits(self.ctrl_limits, limit_type="control")

        self.g: List[ca.SX] = []
        self._init_decision_variables()
        self._check_correct_dimensions(self.X, self.U)
        self.define_bound_constraints()
        self.set_dynamic_constraints()

    def _check_correct_dimensions(self, x: ca.MX, u: ca.MX) -> None:
        """
        Check that the dimensions of the states and controls are correct
        """
        if x.size()[0] != self.casadi_model.n_states:
            raise ValueError("The states do not have the correct dimensions.")

        if u.size()[0] != self.casadi_model.n_controls:
            raise ValueError(
                "The controls do not have the correct dimensions.")

    def _init_decision_variables(self) -> None:
        """
        Initialize the decision variables
        """
        self.X: ca.MX = ca.MX.sym('X', self.casadi_model.n_states, self.N+1)
        self.U: ca.MX = ca.MX.sym('U', self.casadi_model.n_controls, self.N)

        # column vectors for initial and final states
        self.P = ca.MX.sym(
            'P', self.casadi_model.n_states + self.casadi_model.n_states)

        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),
            self.U.reshape((-1, 1)),
        )

    def define_bound_constraints(self):
        """define bound constraints of system"""
        self.variables_list = [self.X, self.U]
        self.variables_name = ['X', 'U']

        # function to turn decision variables into one long row vector
        self.pack_variables_fn = ca.Function(
            'pack_variables_fn', self.variables_list,
            [self.OPT_variables], self.variables_name, ['flat'])

        # function to turn decision variables into respective matrices
        self.unpack_variables_fn = ca.Function(
            'unpack_variables_fn', [self.OPT_variables],
            self.variables_list, ['flat'], self.variables_name)

        # helper functions to flatten and organize constraints
        self.lbx = self.unpack_variables_fn(flat=-ca.inf)
        self.ubx = self.unpack_variables_fn(flat=ca.inf)
        print('Bound constraints defined')

    def update_bound_constraints(self) -> None:
        """
        Users must implement this method to update the bound constraints 
        for the optimization problem
        """
        # num_ctrls = self.casadi_model.n_controls
        # get keys from the control limits dictionary
        ctrl_keys = list(self.ctrl_limits.keys())

        for i, ctrl_name in enumerate(ctrl_keys):
            # get the control limits
            self.lbx['U'][i, :] = self.ctrl_limits[ctrl_name]['min']
            self.ubx['U'][i, :] = self.ctrl_limits[ctrl_name]['max']

        # get keys from the state limits dictionary
        state_keys = list(self.state_limits.keys())
        for i, state_name in enumerate(state_keys):
            # get the state limits
            self.lbx['X'][i, :] = self.state_limits[state_name]['min']
            self.ubx['X'][i, :] = self.state_limits[state_name]['max']

    @abstractmethod
    def compute_total_cost(self) -> ca.MX:
        """
        Users must implement this method to compute 
        the total cost of the optimization problem
        """
        return

    @abstractmethod
    def solve(self, x0: np.ndarray, xF: np.ndarray, u0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Users must implement this method to solve the optimization problem
        """
        return

    def set_dynamic_constraints(self):
        """set dynamic constraints of system"""
        self.g = self.X[:, 0] - self.P[:self.casadi_model.n_states]
        for k in range(self.N):
            # state_next_rk4 = self.integrator(self.X[:, k], self.U[:, k])
            # state_next_rk4 = self.integrator_fn(self.X[:, k], self.U[:, k])
            states = self.X[:, k]
            controls = self.U[:, k]
            k1 = self.casadi_model.function(states, controls)
            k2 = self.casadi_model.function(states + self.dt/2 * k1, controls)
            k3 = self.casadi_model.function(states + self.dt/2 * k2, controls)
            k4 = self.casadi_model.function(states + self.dt * k3, controls)
            state_next_rk4 = states + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            # constraint to make sure our dynamics are satisfied
            self.g = ca.vertcat(self.g, self.X[:, k+1] - state_next_rk4)

    def init_solver(self, cost_fn: ca.MX, solver_opts: Dict = None) -> None:
        """initialize the solver"""
        nlp_prob = {
            'f': cost_fn,
            'x': self.OPT_variables,
            'g': self.g,
            'p': self.P
        }

        if solver_opts is None:
            solver_opts = {
                'ipopt': {
                    # 'max_iter': 50,
                    # 'max_cpu_time': 0.10,
                    # 'max_wall_time': 0.10,
                    'print_level': 0,
                    'warm_start_init_point': 'yes',  # use the previous solution as initial guess
                    'acceptable_tol': 1e-2,
                    'acceptable_obj_change_tol': 1e-2,
                    # 'hsllib': '/usr/local/lib/libcoinhsl.so', #need to set the optimizer library
                    # 'hsllib': '/usr/local/lib/libfakemetis.so', #need to set the optimizer library
                    # 'linear_solver': 'ma57',
                    # 'hessian_approximation': 'limited-memory', # Changes the hessian calculation for a first order approximation.
                },
                # 'verbose': True,
                # 'jit':True,
                'print_time': 0,
                'expand': 1
            }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, solver_opts)
        print('Solver initialized')

    def init_optimization(self, solver_opts: Dict = None) -> None:
        """initialize the optimization problem"""
        if self.is_initialized:
            self.g = []
            self.cost: float = 0.0

        self.update_bound_constraints()
        self.cost = self.compute_total_cost()
        self.init_solver(self.cost, solver_opts)
        self.is_initialized = True

    def unpack_solution(self, sol: Dict) -> Tuple[np.ndarray, np.ndarray]:
        u = ca.reshape(sol['x'][self.casadi_model.n_states * (self.N + 1):],
                       self.casadi_model.n_controls, self.N)
        x = ca.reshape(sol['x'][: self.casadi_model.n_states * (self.N+1)],
                       self.casadi_model.n_states, self.N+1)

        return x, u

    def get_solution(self, solution: Dict) -> Dict:
        """
        Get the solution of the optimization problem
        """
        x, u = self.unpack_solution(solution)
        state_keys = list(self.state_limits.keys())
        ctrl_keys = list(self.ctrl_limits.keys())

        state_dict = {}
        ctrl_dict = {}

        for i, state_name in enumerate(state_keys):
            state_dict[state_name] = x[i, :].full().T[:, 0]

        for i, ctrl_name in enumerate(ctrl_keys):
            ctrl_dict[ctrl_name] = u[i, :].full().T[:, 0]

        return {
            "states": state_dict,
            "controls": ctrl_dict
        }

    def solve_and_get_solution(self, x0: np.ndarray,
                               xF: np.ndarray,
                               u0: np.ndarray) -> Dict:
        """
        Solve the optimization problem and get the solution
        """
        # if not self.is_initialized:
        #     self.init_optimization()

        solution = self.solve(x0, xF, u0)
        return self.get_solution(solution)
