import casadi as ca
from optitraj.models.casadi_model import CasadiModel


class JSBPlane(CasadiModel):
    def __init__(self,
                 dt_val: float = 0.1,
                 pitch_tau: float = 0.5) -> None:
        super().__init__()
        self.dt_val: float = dt_val
        self.pitch_tau = pitch_tau
        self.define_states()
        self.define_controls()
        self.define_state_space()

    def define_states(self) -> None:
        # positions ofrom world
        self.x_f = ca.MX.sym('x_f')
        self.y_f = ca.MX.sym('y_f')
        self.z_f = ca.MX.sym('z_f')

        # attitude
        self.phi_f = ca.MX.sym('phi_f')
        self.theta_f = ca.MX.sym('theta_f')
        self.psi_f = ca.MX.sym('psi_f')
        self.v = ca.MX.sym('t')

        self.states = ca.vertcat(
            self.x_f,
            self.y_f,
            self.z_f,
            self.phi_f,
            self.theta_f,
            self.psi_f,
            self.v
        )

        self.n_states = self.states.size()[0]  # is a column vector

    def define_controls(self) -> None:
        """"""
        self.u_psi = ca.MX.sym('u_psi')
        self.u_z = ca.MX.sym('u_z')
        self.v_cmd = ca.MX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_psi,
            self.u_z,
            self.v_cmd
        )

        self.n_controls = self.controls.size()[0]

    def define_state_space(self) -> None:
        self.g = 9.81
        self.x_fdot = self.v_cmd * ca.cos(self.psi_f) * ca.cos(self.theta_f)
        self.y_fdot = self.v_cmd * ca.sin(self.psi_f) * ca.cos(self.theta_f)
        self.z_fdot = self.v_cmd * ca.sin(self.theta_f)  # + self.u_z

        self.phi_fdot = self.phi_f
        self.theta_fdot = self.theta_f
        self.psi_fdot = self.u_psi * (1/self.pitch_tau)
        self.v_dot = ca.sqrt(self.x_fdot**2 + self.y_fdot**2 + self.z_fdot**2)

        self.z_dot = ca.vertcat(
            self.x_fdot,
            self.y_fdot,
            self.z_fdot,
            self.phi_fdot,
            self.theta_fdot,
            self.psi_fdot,
            self.v_dot
        )

        name = 'dynamics'
        self.function = ca.Function(name,
                                    [self.states, self.controls],
                                    [self.z_dot])


class Plane(CasadiModel):
    def __init__(self,
                 dt_val: float = 0.1,
                 airspeed_tau: float = 0.05,
                 pitch_tau: float = 0.02) -> None:
        super().__init__()
        self.dt_val = dt_val
        self.airspeed_tau = airspeed_tau
        self.pitch_tau = pitch_tau
        self.define_states()
        self.define_controls()
        self.define_state_space()

    def define_states(self) -> None:
        """define the states of your system"""
        # positions ofrom world
        self.x_f = ca.MX.sym('x_f')
        self.y_f = ca.MX.sym('y_f')
        self.z_f = ca.MX.sym('z_f')

        # attitude
        self.phi_f = ca.MX.sym('phi_f')
        self.theta_f = ca.MX.sym('theta_f')
        self.psi_f = ca.MX.sym('psi_f')
        self.v = ca.MX.sym('t')

        self.states = ca.vertcat(
            self.x_f,
            self.y_f,
            self.z_f,
            self.phi_f,
            self.theta_f,
            self.psi_f,
            self.v
        )

        self.n_states = self.states.size()[0]  # is a column vector

    def define_controls(self) -> None:
        """controls for your system"""
        self.u_phi = ca.MX.sym('u_phi')
        self.u_theta = ca.MX.sym('u_theta')
        self.u_psi = ca.MX.sym('u_psi')
        self.v_cmd = ca.MX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_psi,
            self.v_cmd
        )
        self.n_controls = self.controls.size()[0]

    def define_state_space(self) -> None:
        """define the state space of your system"""
        self.g = 9.81  # m/s^2
        # #body to inertia frame
        self.x_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.cos(self.psi_f)
        self.y_fdot = self.v_cmd * ca.cos(self.theta_f) * ca.sin(self.psi_f)
        self.z_fdot = -self.v_cmd * ca.sin(self.theta_f)
        # self.x_fdot = self.v_cmd *  ca.sin(-self.psi_f)
        # self.y_fdot = self.v_cmd *  ca.cos(-self.psi_f)
        # self.z_fdot = -self.v_cmd * ca.sin(self.theta_f)

        self.phi_fdot = -self.u_phi * (1/self.pitch_tau) - self.phi_f
        self.theta_fdot = -self.u_theta * 1/0.5 - self.theta_f
        self.v_dot = ca.sqrt(self.x_fdot**2 + self.y_fdot **
                             2 + self.z_fdot**2)
        # -self.g * (ca.tan(self.phi_f) / self.v_cmd)
        self.psi_fdot = self.u_psi

        self.z_dot = ca.vertcat(
            self.x_fdot,
            self.y_fdot,
            self.z_fdot,
            self.phi_fdot,
            self.theta_fdot,
            self.psi_fdot,
            self.v_dot
        )

        # ODE function
        name = 'dynamics'
        self.function = ca.Function(name,
                                    [self.states, self.controls],
                                    [self.z_dot])

    def define_state_limits(self) -> None:
        """define the state limits of your system"""
        pass
