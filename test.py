import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from aircraftsim import SimInterface,  HighControlInputs, AircraftIC, DataVisualizer
from optitraj.dynamics_adapter import JSBSimAdapter, AircraftState
from optitraj.models.plane import JSBPlane

"""
Run the aircraft sim and save the data
compare the dynamics of the aircraft sim to the dynamics of the optitraj model
"""


def compute_los(goal_x: float, goal_y: float, sim: SimInterface):
    current_state: AircraftState = sim.get_states()
    dy = goal_x - current_state.x
    dx = goal_y - current_state.y
    los = np.arctan2(dy, dx)

    return los


def f(x, u):
    x_f, y_f, z_f, phi_f, theta_f, psi_f, v = x
    u_phi, u_z, v_cmd = u

    g = 9.81
    x_fdot = v_cmd * np.cos(psi_f) * np.cos(theta_f)
    y_fdot = v_cmd * np.sin(psi_f) * np.cos(theta_f)
    z_fdot = u_z - z_f

    phi_fdot = u_phi - phi_f
    theta_fdot = theta_f
    psi_fdot = g * (np.tan(phi_f) / v_cmd)
    v_dot = v

    x_dot = np.array([x_fdot, y_fdot, z_fdot, phi_fdot,
                     theta_fdot, psi_fdot, v_dot])
    return x_dot


def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def main() -> None:
    jsb_plane = JSBPlane()
    init_cond = [0, 0, 50, 0, 0, 0, 20]
    ac_init_cond = AircraftIC(
        x=init_cond[0], y=init_cond[1], z=init_cond[2],
        roll=init_cond[3],
        pitch=init_cond[4],
        yaw=init_cond[5],
        airspeed_m=20)

    sim = SimInterface(
        aircraft_name='x8',
        init_cond=ac_init_cond,
        sim_freq=60
    )

    N = 2000
    roll_cmd_list = []
    pitch_cmd_list = []
    height_cmd_list = []
    airspeed_cmd_list = []

    roll_cmd = np.deg2rad(-15)
    height = 35
    airspeed = 20
    goal_x = 100
    goal_y = 100

    for i in range(N):

        los = compute_los(goal_x, goal_y, sim)
        current_yaw = sim.get_states().yaw
        error = los - current_yaw
        print("Error: ", np.rad2deg(error))
        # wrap the error to -pi to pi
        if error > np.pi:
            error = error - 2*np.pi
        elif error < -np.pi:
            error = error + 2*np.pi
        roll_cmd = np.arctan2(error*20, -9.81)
        roll_cmd = np.clip(roll_cmd, -np.deg2rad(35), np.deg2rad(35))

        roll_cmd_list.append(roll_cmd)
        pitch_cmd_list.append(np.deg2rad(0))
        height_cmd_list.append(height)
        airspeed_cmd_list.append(airspeed)

        sim.step(HighControlInputs(
            ctrl_idx=0,
            pitch=np.deg2rad(0),
            alt_ref_m=height,
            roll=roll_cmd,
            heading_ref_deg=0,
            yaw=np.deg2rad(0),
            vel_cmd=airspeed
        ))

    # save the data
    jsb_sim_report = sim.report

    data_vis = DataVisualizer(jsb_sim_report)
    x = jsb_sim_report.x
    y = jsb_sim_report.y
    z = jsb_sim_report.z
    phi = jsb_sim_report.roll_dg
    theta = jsb_sim_report.pitch_dg
    psi = jsb_sim_report.yaw_dg
    v = jsb_sim_report.airspeed
    phi_rad = np.deg2rad(phi)
    theta_rad = np.deg2rad(theta)
    psi_rad = np.deg2rad(psi)

    calc_x = []
    calc_y = []
    calc_z = []
    calc_phi = []
    calc_theta = []
    calc_psi = []
    calc_v = []

    sim_freq = 200
    dt = 1/60
    # dt = 0.1
    for i in range(1, N):
        x_f, y_f, z_f, phi_f, theta_f, psi_f, v_f = init_cond[0], init_cond[
            1], init_cond[2], init_cond[3], init_cond[4], init_cond[5], init_cond[6]
        u_phi, u_z, v_cmd = roll_cmd_list[i], height_cmd_list[i], airspeed_cmd_list[i]
        next_x = np.array([x_f, y_f, z_f, phi_f, theta_f, psi_f, v_f])
        u = np.array([u_phi, u_z, v_cmd])
        x_next = rk4_step(f, next_x, u, dt)
        x_f, y_f, z_f, phi_f, theta_f, psi_f, v_f = x_next
        # wrap the yaw between 0 to 2pi
        if psi_f > 2*np.pi:
            psi_f -= 2*np.pi
        elif psi_f < 0:
            psi_f += 2*np.pi
        init_cond = x_next
        calc_x.append(y_f)
        calc_y.append(x_f)
        calc_z.append(z_f)
        calc_phi.append(phi_f)
        calc_theta.append(theta_f)
        calc_psi.append(psi_f)
        calc_v.append(v_f)

    fig, ax = data_vis.plot_3d_trajectory()
    ax.plot(calc_x, calc_y, calc_z, label='calculated', color='r')
    ax.scatter(calc_x[0], calc_y[0], calc_z[0], label='start', color='g')
    ax.scatter(goal_x, goal_y, 0, label='goal')
    ax.legend()

    fig, ax = data_vis.plot_attitudes()
    time_vector = jsb_sim_report.time[1:]
    ax[0].plot(time_vector, np.rad2deg(calc_phi),
               label='calculated', color='r')
    ax[1].plot(time_vector, np.rad2deg(calc_theta),
               label='calculated', color='g')
    ax[2].plot(time_vector, np.rad2deg(calc_psi),
               label='calculated', color='g', linestyle='--')

    plt.show()


if __name__ == '__main__':
    main()
