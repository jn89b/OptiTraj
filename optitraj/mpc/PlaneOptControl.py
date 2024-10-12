
import numpy as np
import casadi as ca

from casadi.casadi import MX
from optitraj.models.casadi_model import CasadiModel
from optitraj.mpc.optimization import OptimalControlProblem
from optitraj.utils.data_container import MPCParams
# from optitraj.models.plane import Plane
# from optitraj.utils.limits import Limits, validate_limits


class PlaneOptControl(OptimalControlProblem):
    """
    Example of a class that inherits from OptimalControlProblem

    """

    def __init__(self,
                 mpc_params: MPCParams,
                 casadi_model: CasadiModel,
                 use_obs_avoidance: bool = False,
                 obs_params: dict = None) -> None:
        super().__init__(mpc_params,
                         casadi_model)

        self.use_obs_avoidance: bool = use_obs_avoidance
        self.obs_params: dict = obs_params
        # self.state_limits = state_limits
        # self.ctrl_limits = ctrl_limits

    def compute_dynamics_cost(self) -> MX:
        """
        Compute the dynamics cost for the optimal control problem
        """
        # initialize the cost
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

    def solve(self, x0: np.ndarray, xF: np.ndarray, u0: np.ndarray) -> np.ndarray:
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
            # set the obstacle avoidance constraints
            num_constraints = n_states*(self.N+1)
            lbg = ca.DM.zeros((num_constraints, 1))
            ubg = ca.DM.zeros((num_constraints, 1))
        else:
            num_constraints = n_states*(self.N+1)
            lbg = ca.DM.zeros((num_constraints, 1))
            ubg = ca.DM.zeros((num_constraints, 1))

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
