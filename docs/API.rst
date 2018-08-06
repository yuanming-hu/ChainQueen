API
==================================

Initialization
---------------------
The initial states of a simulation consist of particle positions and velocities (by default, zero).

Particle configuration
Boundary conditions


Interacting with the simulation states
----------------------------------

A simulation consists of a series of states, one per time step.
You can get the (symbolic) simulation from `sim.states`

.. code-block:: python

    from simulation import Simulation
    sim = Simulation(sess, res=(25, 25))
    state = sim.states
    # Particle Positions
    # Array of float32[batch, particle, dimension=0,1]
    state['position']

    # Particle Velocity
    # Array of float32[batch, particle, dimension=0,1]
    state['velocity'] # Array of float32[batch, particle, dimension=0,1]

    # Particle Deformation Gradients
    # Array of float32[batch, particle, matrix dimension0=0,1, matrix dimension1=0,1]
    state['deformation_gradient']
