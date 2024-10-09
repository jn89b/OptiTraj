import pickle as pkl
import numpy as np

with open('next_state.pkl', 'rb') as f:
    next_state = pkl.load(f)
    
with open('states.pkl', 'rb') as f:
    state = pkl.load(f)
    

x_current = state['x']
y_current = state['y']
z_current = state['z']
phi_current = state['phi']
theta_current = state['theta']
psi_current = state['psi']
v_current = state['v']


