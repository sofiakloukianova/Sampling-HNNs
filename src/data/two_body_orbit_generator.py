# Adapterd from :
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import scipy
from hamiltonian import BaseHamiltonian

solve_ivp = scipy.integrate.solve_ivp

import os, sys, pickle

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

##### 2-BODY ORBITS #####
def random_config(orbit_noise=5e-2, min_radius=0.5, max_radius=1.5):
    state = np.zeros((2, 5))
    state[:, 0] = 1
    pos = np.random.rand(2) * (max_radius - min_radius) + min_radius
    r = np.sqrt(np.sum((pos ** 2)))

    # velocity that yields a circular orbit
    vel = np.flipud(pos) / (2 * r ** 1.5)
    vel[0] *= -1
    vel *= 1 + orbit_noise * (2 * np.random.rand() - 1)

    # make the circular orbits SLIGHTLY elliptical
    # For body 0: position = pos, velocity = vel.
    # For body 1: they just flip the sign of everything except the mass.
    state[:, 1:3] = pos
    state[:, 3:5] = vel
    state[1, 1:] *= -1
    return state

##### DATASET SAVING/LOADING #####

def to_pickle(thing, path): # save something
    with open(path, 'wb') as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path): # load something
    thing = None
    with open(path, 'rb') as handle:
        thing = pickle.load(handle)
    return thing

##### ENERGY #####
def potential_energy(state):
    '''U=sum_i,j>i G m_i m_j / r_ij'''
    tot_energy = np.zeros((1, 1, state.shape[2]))
    for i in range(state.shape[0]):
        for j in range(i + 1, state.shape[0]):
            r_ij = ((state[i:i + 1, 1:3] - state[j:j + 1, 1:3]) ** 2).sum(1, keepdims=True) ** .5
            m_i = state[i:i + 1, 0:1]
            m_j = state[j:j + 1, 0:1]
    tot_energy += m_i * m_j / r_ij
    U = -tot_energy.sum(0).squeeze()
    return U

def kinetic_energy(state):
    '''T=sum_i .5*m*v^2'''
    energies = .5 * state[:, 0:1] * (state[:, 3:5] ** 2).sum(1, keepdims=True)
    T = energies.sum(0).squeeze()
    return T

def total_energy(state):
    return potential_energy(state) + kinetic_energy(state)


##### DYNAMICS #####
def get_accelerations(state, epsilon=0):
    # shape of state is [bodies x properties]
    net_accs = []  # [nbodies x 2]
    for i in range(state.shape[0]):  # number of bodies
        other_bodies = np.concatenate([state[:i, :], state[i + 1:, :]], axis=0)
        displacements = other_bodies[:, 1:3] - state[i, 1:3]  # indexes 1:3 -> pxs, pys
        distances = (displacements ** 2).sum(1, keepdims=True) ** 0.5
        masses = other_bodies[:, 0:1]  # index 0 -> mass
        pointwise_accs = masses * displacements / (distances ** 3 + epsilon)  # G=1
        net_acc = pointwise_accs.sum(0, keepdims=True)
        net_accs.append(net_acc)
    net_accs = np.concatenate(net_accs, axis=0)
    return net_accs


def update(t, state):
    state = state.reshape(-1, 5)  # [bodies, properties]
    deriv = np.zeros_like(state)
    deriv[:, 1:3] = state[:, 3:5]  # dx, dy = vx, vy
    deriv[:, 3:5] = get_accelerations(state)
    return deriv.reshape(-1)


##### INTEGRATE ORBITS / GENERATE ORBIT DATASET #####
def get_orbit(state, update_fn=update, t_points=100, t_span=[0, 2], **kwargs):
    if not 'rtol' in kwargs.keys():
        kwargs['rtol'] = 1e-9

    orbit_settings = locals()

    nbodies = state.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    orbit_settings['t_eval'] = t_eval

    path = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(),
                     t_eval=t_eval, **kwargs)
    orbit = path['y'].reshape(nbodies, 5, t_points)
    return orbit, orbit_settings


def sample_orbits(train_size, test_size, nbodies=2, orbit_noise=5e-2,
                  min_radius=0.5, max_radius=1.5, t_span=[0, 20], verbose=False, **kwargs):

    steps_target = 20
    trials, timesteps = compute_orbit_config(train_size, test_size, steps_target)

    # For train=10000 and test=2000, we get 20 timesteps and 600 trials
    print(f"Trials: {trials}, Timesteps: {timesteps}")

    orbit_settings = locals()
    if verbose:
        print("Making a dataset of near-circular 2-body orbits:")

    x, dx, e = [], [], []
    N = timesteps * trials
    while len(x) < N:

        state = random_config(orbit_noise, min_radius, max_radius)
        orbit, settings = get_orbit(state, t_points=timesteps, t_span=t_span, **kwargs)
        batch = orbit.transpose(2, 0, 1).reshape(-1, 10)

        for state in batch:
            dstate = update(None, state)

            # reshape from [nbodies, state] where state=[m, qx, qy, px, py]
            # to [canonical_coords] = [qx1, qx2, qy1, qy2, px1,px2,....]
            coords = state.reshape(nbodies, 5).T[1:].flatten()
            dcoords = dstate.reshape(nbodies, 5).T[1:].flatten()
            x.append(coords)
            dx.append(dcoords)

            shaped_state = state.copy().reshape(2, 5, 1)
            e.append(total_energy(shaped_state))

    data = {'coords': np.stack(x)[:N],
            'dcoords': np.stack(dx)[:N],
            'energy': np.stack(e)[:N]}

    return data, orbit_settings

def compute_orbit_config(train_size, test_size, steps_target=20):
    """
    Dynamically compute number of initial conditions (ICs) and time steps per orbit.
    We try to match roughly ICs * steps ≈ dataset_size.
    """
    total_needed = train_size + test_size # target ~ steps_target steps per orbit
    num_steps = steps_target
    num_orbits = int(np.ceil(total_needed / num_steps)) # choose number of orbits to generate

    print(f"[Config] num_orbits={num_orbits}, num_steps={num_steps}, total={num_orbits * num_steps}")
    return num_orbits, num_steps


##### MAKE A DATASET #####
def make_orbits_dataset(train_size=10000, test_size=2000, **kwargs):
    data, orbit_settings = sample_orbits(train_size, test_size, **kwargs)

    print(f"Fulll len - {len(data['coords'])}")
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[:train_size], v[train_size:train_size + test_size]
    data = split_data
    print(f"Train len - {len(data['coords'])}")
    print(f"Test len - {len(data['test_coords'])}")

    data['meta'] = orbit_settings
    return data


##### LOAD OR SAVE THE DATASET #####
def get_dataset(target: BaseHamiltonian, experiment_name, save_dir,
                train_size, test_size, rng, **kwargs):
    '''Returns an orbital dataset. Also constructs
    the dataset if no saved version is available.'''
    os.makedirs(save_dir, exist_ok=True)

    path = '{}/{}-orbits-dataset.pkl'.format(save_dir, experiment_name)
    #data = make_orbits_dataset(**kwargs)
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        #to_pickle(data, path)

    data = make_orbits_dataset(train_size=train_size, test_size=test_size, **kwargs)

    return data


##### HELPER FUNCTION #####
def coords2state(coords, nbodies=2, mass=1):
    timesteps = coords.shape[0]
    state = coords.T
    state = state.reshape(-1, nbodies, timesteps).transpose(1, 0, 2)
    mass_vec = mass * np.ones((nbodies, 1, timesteps))
    state = np.concatenate([mass_vec, state], axis=1)
    return state
