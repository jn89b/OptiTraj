"""
Define System Dynamics with casadi_model
Set the weights of your matrices

Create the Optimal Control Problem by inheriting from OptimalControlProblem
- Define the dynamics cost
- Define the total cost
- Define the solve method

Define the state and control limits
Define the MPC parameters

If you have a custom dynamics simulator, you can use the dynamics adapter to interface with the simulator

Define the initial and final states
Define the initial controls
Define the custom stop criteria


"""
