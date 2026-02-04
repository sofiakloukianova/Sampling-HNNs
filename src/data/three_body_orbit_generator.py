# Adapted from :
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import numpy as np
import scipy
from hamiltonian import BaseHamiltonian
from integrator.implicit_midpoint import implicit_midpoint
import torch

solve_ivp = scipy.integrate.solve_ivp

import os, sys, pickle

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

##### 3-BODY ORBITS #####

def figure8_config(scale=1.0):
    """
    Periodic figure-8 initial condition (Newtonian 3-body).
    Reference: A. Chenciner, R. Montgomery
    """
    state = np.zeros((3,5))
    state[:,0] = 1  # equal masses

    # positions
    state[0, 1:3] = scale * np.array([-1.0,  0.0])
    state[1, 1:3] = scale * np.array([ 1.0,  0.0])
    state[2, 1:3] = scale * np.array([0.0, 0.0])

    # velocities
    state[0, 3:5] = scale * np.array([ 0.347113,  0.532727])
    state[1, 3:5] = scale * np.array([ 0.347113,  0.532727])
    state[2, 3:5] = scale * np.array([-0.694226, -1.065454])
    return state

def brouckeA1_config(scale=1.0):
    """
    Periodic Broucke A1 initial condition (Newtonian 3-body).
    Reference: R. Broucke, D. Boggs
    """
    state = np.zeros((3, 5))
    state[:, 0] = 1  # equal masses

    # positions
    state[0, 1:3] = scale * np.array([-0.9892620043, 0.0])
    state[1, 1:3] = scale * np.array([ 2.2096177241, 0.0])
    state[2, 1:3] = scale * np.array([-1.2203557197, 0.0])

    # velocities
    state[0, 3:5] = scale * np.array([0.0,  1.9169244185])
    state[1, 3:5] = scale * np.array([0.0,  0.1910268738])
    state[2, 3:5] = scale * np.array([0.0, -2.1079512924])
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
    '''U=\sum_i,j>i G m_i m_j / r_ij'''
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
    '''T=\sum_i .5*m*v^2'''
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

def update_np(x, model):
    # Compute time derivative for S-HNN for ODE integration.
    if getattr(model, "is_torch_model", False):
        # HNN, MLP : expects torch.Tensor input
        x_tensor = torch.tensor(x[None, :], dtype=torch.float64, requires_grad=True)
        dx_dt = model.dt(x_tensor).detach().cpu().numpy()
        return dx_dt[0]
    else:
        # S-HNN, S-MLP : expects numpy input directly
        x_batch = x[None, :]  # shape (1, D)
        dx_dt = model.dt(x_batch)
        return dx_dt[0]

def update_coords(x):
    """
    x: torch tensor shape (1,12)
    returns dx/dt as (1,12)
    """
    x_np = x.detach().cpu().numpy().reshape(12,) # --- convert torch -> numpy (12,) ---
    state = coords_to_state_3body(x_np) # --- coords -> state (3,5) ---

    # --- compute derivative in state space ---
    deriv_flat = update(0, state)
    deriv = deriv_flat.reshape(3,5)

    dcoords = state_to_coords_3body(deriv)  # --- state -> coords (12,) ---
    return torch.tensor(dcoords[None, :], dtype=torch.float64)

def update_torch(x, model):
    if getattr(model, "is_torch_model", False):
        # HNN, MLP → expects torch.Tensor input
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        dx_dt = model.dt(x)
        return dx_dt
    else:
        # S-HNN → expects numpy.ndarray
        x_np = x.detach().cpu().numpy()
        dx_dt = model.dt(x_np)
        return torch.from_numpy(dx_dt).to(x.device)


##### INTEGRATE ORBITS / GENERATE ORBIT DATASET #####
def get_orbit(state, update_fn=update_coords, t_points=100, t_span=[0, 2], nbodies=3, **kwargs):
    if not 'rtol' in kwargs.keys():
        kwargs['rtol'] = 1e-9

    orbit_settings = locals()

    nbodies = state.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    orbit_settings['t_eval'] = t_eval

    x0 = state_to_coords_3body(state)  # (12,)
    x0 = torch.tensor(x0[None, :], dtype=torch.float64)
    num_steps = t_points - 1
    trajectory_torch = implicit_midpoint(x0, update_fn, dt=(t_span[1] - t_span[0]) / num_steps, num_steps=num_steps)
    trajectory = trajectory_torch[0].cpu().numpy()

    orbit = coords2state_3body(trajectory)
    return orbit, orbit_settings


def sample_orbits(target: BaseHamiltonian, train_size, test_size, rng=None, orbit_noise=0.02,
                  t_span=[0, 5], steps_target=200, verbose=False):
    """
    Generate raw coordinates, then recompute derivatives & energy from Hamiltonian.
    """
    if verbose:
        print(f"[three-body] Sampling {train_size + test_size} chaotic orbits...")


    num_init, num_steps = compute_orbit_config(train_size, test_size, steps_target)

    state_fig8 = figure8_config()
    orbit_fig8, settings_fig8 = get_orbit(state_fig8, t_points=1000, t_span=[0, 3])

    #state_fig8 = brouckeA1_config()
    #orbit_fig8, settings_fig8 = get_orbit(state_fig8, t_points=1000, t_span=[0, 7])

    idx = np.linspace(0, 999, num_init, dtype=int)

    # === 2) extract x,y,vx,vy from these timesteps ===
    x0_clean = np.zeros((num_init, 3, 4))
    for i, t in enumerate(idx):
        x0_clean[i] = orbit_fig8[:, 1:5, t]  # no mass here ON PURPOSE

    # === 3) perturb positions (not velocities) ===
    x0 = x0_clean.copy()
    x0[:, :, :2] += orbit_noise * rng.standard_normal((num_init, 3, 2))

    # === storage ===
    coords_list = []
    dcoords_list = []
    energy_list = []

    # helper: add mass column
    def add_mass(x):
        out = np.zeros((3, 5))
        out[:, 0] = 1
        out[:, 1:5] = x
        return out

    # === 4) integrate each noisy init conditins ===
    for i in range(num_init):
        init_state_5 = add_mass(x0[i])  # convert (3,4) -> (3,5)

        # integrate chaotic rollout
        orbit_i, _ = get_orbit(
            init_state_5,
            t_points=num_steps,
            t_span=t_span
        )
        # remove mass column → (num_steps,3,4)
        traj = orbit_i[:, 1:5, :].transpose(2, 0, 1)

        # === HNN coords ===
        q_flat = traj[:, :, 0:2].reshape(num_steps, -1)
        p_flat = traj[:, :, 2:4].reshape(num_steps, -1)

        # coords_i[t] = [qx1,qy1, qx2,qy2 ,qx3,qy3, px1,py1, px2,py2, px3,py3]t​
        coords_i = np.concatenate([q_flat, p_flat], axis=1)  # (num_steps,12)
        dcoords_i = target.dt(coords_i)
        energy_i = target.H(coords_i)[:, 0]

        coords_list.append(coords_i)
        dcoords_list.append(dcoords_i)
        energy_list.append(energy_i)

    coords = np.concatenate(coords_list, axis=0)
    dcoords = np.concatenate(dcoords_list, axis=0)
    energy = np.concatenate(energy_list, axis=0)

    # shuffle
    perm = rng.permutation(len(coords))
    coords = coords[perm]
    dcoords = dcoords[perm]
    energy = energy[perm]

    return {
        "x0_clean": x0_clean,  # reference clean figure-8 samples
        "x0_noisy": x0,  # (num_init,3,4)
        "coords": coords,  # (total_size,12)
        "dcoords": dcoords,  # (total_size,12)
        "energy": energy,  # (total_size,)
        "idx": idx  # which timesteps were used
    }

def compute_orbit_config(train_size, test_size, steps_target=200):
    """
    Dynamically compute number of initial conditions (ICs) and time steps per orbit.
    We try to match roughly ICs * steps ≈ dataset_size.
    """
    total_needed = train_size + test_size
    num_steps = steps_target # target ~ steps_target steps per orbit
    num_orbits = int(np.ceil(total_needed / num_steps)) # choose number of orbits to generate

    print(f"[Config] num_orbits={num_orbits}, num_steps={num_steps}, total={num_orbits * num_steps}")
    return num_orbits, num_steps


##### MAKE A DATASET #####
def make_orbits_dataset(target: BaseHamiltonian, train_size=10000, test_size=2000, rng=None, **kwargs):
    data = sample_orbits(target, train_size, test_size, rng, **kwargs)

    print(f"Full len {len(data['coords'])}")
    split_data = {}
    for k, v in data.items():
        split_data[k], split_data['test_' + k] = v[:train_size], v[train_size:train_size + test_size]
    data = split_data
    print(f"Train len - {len(data['coords'])}")
    print(f"Test len - {len(data['test_coords'])}")

    return data


##### LOAD OR SAVE THE DATASET #####
def get_dataset(target: BaseHamiltonian, experiment_name, save_dir,
                train_size, test_size, rng, **kwargs):
    '''Returns an orbital dataset. Also constructs
    the dataset if no saved version is available.'''
    os.makedirs(save_dir, exist_ok=True)
    path = '{}/{}-orbits-dataset.pkl'.format(save_dir, experiment_name)

    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        #
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        #to_pickle(data, path)
    data = make_orbits_dataset(target, train_size=train_size, test_size=test_size, rng=rng, **kwargs)
    return data


##### CONVERSIONS #####
def state_to_coords_3body(state):
    """
    Convert state (3,5): [m, x, y, vx, vy]
    into coords (12): [qx1,qx2,qx3, qy1,qy2,qy3, px1,px2,px3, py1,py2,py3]
    """
    state = np.asarray(state)
    m = state[:,0]
    qx = state[:,1]
    qy = state[:,2]
    px = state[:,3]
    py = state[:,4]

    coords = np.concatenate([qx, qy, px, py], axis=0)
    return coords

def coords2state_3body(traj):
    """
    traj: array, shape(T, 12): rows [qx1,qx2,qx3, qy1,qy2,qy3, px1,px2,px3, py1,py2,py3]
    returns: array, shape (3,5,T) : rows [m, x, y, px, py]
    """
    T = traj.shape[0]
    qx1, qx2, qx3, qy1, qy2, qy3 = traj[:, 0:6].T
    px1, px2, px3, py1, py2, py3 = traj[:, 6:12].T
    ones = np.ones(T)

    body1 = np.vstack([ones, qx1, qy1, px1, py1])
    body2 = np.vstack([ones, qx2, qy2, px2, py2])
    body3 = np.vstack([ones, qx3, qy3, px3, py3])

    return np.array([body1, body2, body3])

def xgroup_to_bodygroup(x):
    """
    x : array, shape (12,) :  [qx1,qx2,qx3, qy1,qy2,qy3, px1,px2,px3, py1,py2,py3]
    returns: array, shape (12,) : [x1,y1, x2,y2, x3,y3, px1,py1, px2,py2, px3,py3]
    """
    x1, x2, x3 = x[0:3]
    y1, y2, y3 = x[3:6]
    px1, px2, px3 = x[6:9]
    py1, py2, py3 = x[9:12]
    return np.array([ x1, y1, x2, y2, x3, y3, px1, py1, px2, py2, px3, py3 ])

def bodygroup_to_xgroup(xb):
    """
    x : array, shape (12,) :  [x1,y1, x2,y2, x3,y3, px1,py1, px2,py2, px3,py3]
    returns: array, shape (12,) : [qx1,qx2,qx3, qy1,qy2,qy3, px1,px2,px3, py1,py2,py3]
    """
    return np.array([
        xb[0], xb[2], xb[4], # qx1, qx2, qx3
        xb[1], xb[3], xb[5], # qy1, qy2, qy3
        xb[6], xb[8], xb[10], # px1, px2, px3
        xb[7], xb[9], xb[11] # py1, py2, py3
    ])

def orbit_state_to_coords12(orbit_state):
    """
    orbit_state: (3,5,T) with rows [m, x, y, vx, vy]
    returns (T,12): [qx1,qx2,qx3,qy1,qy2,qy3,px1,px2,px3,py1,py2,py3]
    """
    qx = orbit_state[:, 1, :].T  # (T,3)
    qy = orbit_state[:, 2, :].T  # (T,3)
    px = orbit_state[:, 3, :].T  # (T,3)
    py = orbit_state[:, 4, :].T  # (T,3)
    return np.hstack([qx, qy, px, py])  # (T,12)

def coords_to_state_3body(coords):
    """
    coords: flat (12,) [qx1,qx2,qx3, qy1,qy2,qy3, px1,px2,px3, py1,py2,py3]
    returns state: (3,5) with [m,x,y,vx,vy]
    """
    coords = np.asarray(coords).reshape(12,)
    qx = coords[0:3]
    qy = coords[3:6]
    px = coords[6:9]
    py = coords[9:12]

    state = np.zeros((3,5))
    state[:,0] = 1
    state[:,1] = qx
    state[:,2] = qy
    state[:,3] = px
    state[:,4] = py
    return state
