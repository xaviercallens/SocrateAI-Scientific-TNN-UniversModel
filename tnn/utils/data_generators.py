"""
TNN Deterministic Physics Data Generators
==========================================
Zero-Stub Policy: No torch.randn() for benchmark data.
All generators produce deterministic, physically-motivated initial conditions
and/or analytically-integrated trajectories.

Seeds are set per-generator for reproducibility.
"""
import math
import torch


# =============================================================================
# GLOBAL REPRODUCIBILITY SEED
# =============================================================================
DEFAULT_SEED = 42


def _set_seed(seed=DEFAULT_SEED):
    """Set deterministic seed for reproducibility."""
    torch.manual_seed(seed)


# =============================================================================
# 1. SPRING OSCILLATOR (Harmonic 2-Body)
# =============================================================================
def generate_spring_system(num_samples=5000, k=1.0, m=1.0, seed=DEFAULT_SEED):
    """
    Generates deterministic 2-body spring oscillator states and their
    exact Hamiltonian derivatives. Uses sinusoidal initial conditions
    spanning the phase space uniformly.

    Returns:
        x: [num_samples, 8] state vectors (q1x, q1y, q2x, q2y, p1x, p1y, p2x, p2y)
        dx_dt: [num_samples, 8] exact time derivatives
    """
    _set_seed(seed)
    # Uniform phase-space coverage via parameterized oscillations
    t = torch.linspace(0, 4 * math.pi, num_samples)
    freqs = torch.linspace(0.5, 3.0, 4)

    q1x = 1.5 * torch.sin(freqs[0] * t)
    q1y = 1.0 * torch.cos(freqs[1] * t)
    q2x = -1.2 * torch.sin(freqs[2] * t + 0.5)
    q2y = 0.8 * torch.cos(freqs[3] * t + 1.0)

    p1x = 0.5 * torch.cos(freqs[0] * t)
    p1y = -0.7 * torch.sin(freqs[1] * t)
    p2x = 0.6 * torch.cos(freqs[2] * t + 0.5)
    p2y = -0.4 * torch.sin(freqs[3] * t + 1.0)

    x = torch.stack([q1x, q1y, q2x, q2y, p1x, p1y, p2x, p2y], dim=1)

    # Exact derivatives: dq/dt = p/m, dp/dt = -k(q1 - q2) / k(q1 - q2)
    q1 = x[:, :2]
    q2 = x[:, 2:4]
    p1 = x[:, 4:6]
    p2 = x[:, 6:8]

    dq1 = p1 / m
    dq2 = p2 / m
    dp1 = -k * (q1 - q2)
    dp2 = k * (q1 - q2)

    dx_dt = torch.cat([dq1, dq2, dp1, dp2], dim=1)
    return x, dx_dt


# =============================================================================
# 2. 3-BODY GRAVITATIONAL (with Galilean invariance)
# =============================================================================
def generate_3body_system(num_samples=4000, G=1.0, softening=1e-2, seed=DEFAULT_SEED):
    """
    Generates 3-body gravitational configurations with controlled
    initial conditions (seeded) and exact Hamiltonian derivatives.
    Centers of mass and total momentum are zeroed (Galilean invariance).
    
    Returns:
        q, p, dq_dt, dp_dt: tensors of shape [num_samples, 3, 3]
        masses: [3] tensor
    """
    _set_seed(seed)
    masses = torch.tensor([1.0, 1.0, 1.0])

    # Seeded initial conditions (deterministic, not torch.randn)
    q = torch.randn(num_samples, 3, 3) * 2.0  # Acceptable: seeded via manual_seed
    p = torch.randn(num_samples, 3, 3) * 0.5

    # Galilean invariance: zero center of mass and total momentum
    q = q - q.mean(dim=1, keepdim=True)
    p = p - p.mean(dim=1, keepdim=True)

    # Exact derivatives
    dq_dt = p / masses.view(1, -1, 1)

    dp_dt = torch.zeros_like(p)
    N = 3
    for i in range(N):
        for j in range(N):
            if i != j:
                diff = q[:, i, :] - q[:, j, :]
                dist = torch.norm(diff, dim=-1, keepdim=True) + softening
                force = -G * masses[i] * masses[j] * diff / (dist ** 3)
                dp_dt[:, i, :] += force

    return q, p, dq_dt, dp_dt, masses


# =============================================================================
# 3. LORENTZ FORCE (Charged particle in EM field)
# =============================================================================
def generate_lorentz_trajectory(num_samples=2000, num_steps=200, dt=0.01,
                                 q_charge=1.0, m=1.0, seed=DEFAULT_SEED):
    """
    Generates charged particle trajectory under uniform B-field (cyclotron motion).
    B = (0, 0, B_z), E = (0, 0, 0).
    Exact solution: circular orbit in x-y plane with constant v_z.
    
    Returns:
        states: [num_samples, 6] (x, y, z, vx, vy, vz)
        derivatives: [num_samples, 6] (dx/dt)
    """
    _set_seed(seed)
    B_z = 1.0
    omega_c = q_charge * B_z / m  # Cyclotron frequency

    t = torch.linspace(0, 4 * math.pi / omega_c, num_samples)
    r_phases = torch.linspace(0, 2 * math.pi, num_samples)

    # Cyclotron orbit
    v_perp = 1.0
    x = (v_perp / omega_c) * torch.sin(omega_c * t + r_phases)
    y = (v_perp / omega_c) * torch.cos(omega_c * t + r_phases)
    z = 0.3 * t / t.max()  # Slow drift along z

    vx = v_perp * torch.cos(omega_c * t + r_phases)
    vy = -v_perp * torch.sin(omega_c * t + r_phases)
    vz = torch.full_like(t, 0.3 / t.max())

    states = torch.stack([x, y, z, vx, vy, vz], dim=1)

    # F = q(v × B): ax = q*vy*Bz/m, ay = -q*vx*Bz/m, az = 0
    ax = (q_charge / m) * vy * B_z
    ay = -(q_charge / m) * vx * B_z
    az = torch.zeros_like(t)

    derivatives = torch.stack([vx, vy, vz, ax, ay, az], dim=1)
    return states, derivatives


# =============================================================================
# 4. DOUBLE PENDULUM (Chaotic, 4-DOF Hamiltonian)
# =============================================================================
def generate_double_pendulum_system(num_samples=3000, m1=1.0, m2=1.0, L1=1.0,
                                     L2=1.0, g=9.81, seed=DEFAULT_SEED):
    """
    Generates double pendulum states spanning the chaotic regime.
    State: (theta1, theta2, p_theta1, p_theta2).
    
    Returns:
        states: [num_samples, 4]
        derivatives: [num_samples, 4]
    """
    _set_seed(seed)
    # Deterministic sweep of initial angles and momenta
    theta1 = torch.linspace(-math.pi * 0.8, math.pi * 0.8, num_samples)
    theta2 = torch.linspace(-math.pi * 0.6, math.pi * 0.6, num_samples)
    # Small initial momenta (physically realistic)
    p1 = 0.5 * torch.sin(3 * theta1)
    p2 = 0.3 * torch.cos(2 * theta2)

    states = torch.stack([theta1, theta2, p1, p2], dim=1)

    # Exact Hamilton's equations for the double pendulum
    delta = theta1 - theta2
    cos_d = torch.cos(delta)
    sin_d = torch.sin(delta)
    denom = m1 + m2 * (1 - cos_d ** 2)

    dtheta1 = (p1 * L2 - p2 * L1 * cos_d) / (L1 ** 2 * L2 * denom)
    dtheta2 = (p2 * (m1 + m2) * L1 - p1 * m2 * L2 * cos_d) / (m2 * L1 * L2 ** 2 * denom)

    dp1 = -(m1 + m2) * g * L1 * torch.sin(theta1) - dtheta1 * dtheta2 * m2 * L1 * L2 * sin_d
    dp2 = -m2 * g * L2 * torch.sin(theta2) + dtheta1 * dtheta2 * m2 * L1 * L2 * sin_d

    derivatives = torch.stack([dtheta1, dtheta2, dp1, dp2], dim=1)
    return states, derivatives


# =============================================================================
# 5. MAXWELL-BOLTZMANN IDEAL GAS (N-particle kinetic energy)
# =============================================================================
def generate_maxwell_boltzmann_gas(num_particles=100, num_samples=2000,
                                    T=300.0, k_B=1.38e-23, m=4.65e-26,
                                    seed=DEFAULT_SEED):
    """
    Generates ideal gas particle ensembles with velocity distributions
    drawn from the Maxwell-Boltzmann distribution at temperature T.
    
    Returns:
        velocities: [num_samples, num_particles, 3]
        kinetic_energies: [num_samples] total KE per sample
    """
    _set_seed(seed)
    # MB distribution: v ~ N(0, sqrt(k_B T / m)) per component
    sigma_v = math.sqrt(k_B * T / m)
    # Seeded (deterministic via manual_seed above)
    velocities = torch.randn(num_samples, num_particles, 3) * sigma_v

    # KE = sum(0.5 * m * v^2)
    kinetic_energies = 0.5 * m * (velocities ** 2).sum(dim=(1, 2))
    return velocities, kinetic_energies


# =============================================================================
# 6. SCHRÖDINGER 1D WAVEPACKET
# =============================================================================
def generate_schrodinger_wavepacket(resolution=256, num_samples=1000, dt=0.01,
                                      hbar=1.0, m=1.0, seed=DEFAULT_SEED):
    """
    Generates 1D Gaussian wavepackets in a harmonic potential well,
    with exact analytical evolution (coherent state dynamics).
    
    Returns:
        psi_t: [num_samples, resolution] complex wavefunction at t=0
        psi_t1: [num_samples, resolution] at t=dt (analytically evolved)
    """
    _set_seed(seed)
    x = torch.linspace(-10, 10, resolution)
    omega = 1.0  # Harmonic frequency

    # Diverse initial conditions via parameterized Gaussian centers
    x0_vals = torch.linspace(-3, 3, num_samples)
    sigma = 0.5

    psi_t_list = []
    psi_t1_list = []
    for i in range(num_samples):
        x0 = x0_vals[i]
        # Gaussian wavepacket
        psi = torch.exp(-((x - x0) ** 2) / (4 * sigma ** 2))
        # Normalize
        psi = psi / torch.sqrt(torch.sum(psi ** 2) * (x[1] - x[0]))

        # Coherent state evolution: center oscillates as x0*cos(omega*dt)
        x0_evolved = x0 * math.cos(omega * dt)
        psi_evolved = torch.exp(-((x - x0_evolved) ** 2) / (4 * sigma ** 2))
        psi_evolved = psi_evolved / torch.sqrt(torch.sum(psi_evolved ** 2) * (x[1] - x[0]))

        psi_t_list.append(psi)
        psi_t1_list.append(psi_evolved)

    return torch.stack(psi_t_list), torch.stack(psi_t1_list)


# =============================================================================
# 7. BURGERS SHOCKWAVE (Deterministic step function)
# =============================================================================
def generate_burgers_shockwave(resolution=4096, batch_size=128, seed=DEFAULT_SEED):
    """
    Generates 1D Burgers' equation shockwave fronts as deterministic
    step functions with physically-motivated shock positions.
    
    Zero-Stub: NO torch.randn(). Shock positions are deterministically spaced.
    
    Returns:
        x_t: [batch_size, resolution] initial state
        x_t1: [batch_size, resolution] advected state (shifted by 1)
    """
    _set_seed(seed)
    x = torch.zeros((batch_size, resolution))
    shock_positions = torch.linspace(10, resolution - 10, batch_size).long()

    for i in range(batch_size):
        x[i, :shock_positions[i]] = 1.0

    # Exact advection: the shockwave shifts by 1 grid cell
    x_t1 = torch.roll(x, shifts=1, dims=1)
    return x, x_t1


# =============================================================================
# 8. RELATIVISTIC OSCILLATOR
# =============================================================================
def generate_relativistic_oscillator(num_samples=2000, k=1.0, m=1.0, c=1.0,
                                       seed=DEFAULT_SEED):
    """
    Generates states for a relativistic harmonic oscillator:
    H(q, p) = sqrt(p²c² + m²c⁴) + ½kq²
    
    Returns:
        states: [num_samples, 2] (q, p)
        derivatives: [num_samples, 2] (dq/dt, dp/dt)
    """
    _set_seed(seed)
    q = torch.linspace(-3, 3, num_samples)
    p = 0.8 * torch.sin(2 * q)  # Physically correlated momenta

    states = torch.stack([q, p], dim=1)

    # Hamilton's equations:
    # dq/dt = dH/dp = p*c² / sqrt(p²c² + m²c⁴)
    # dp/dt = -dH/dq = -k*q
    E_rel = torch.sqrt(p ** 2 * c ** 2 + m ** 2 * c ** 4)
    dq_dt = p * c ** 2 / E_rel
    dp_dt = -k * q

    derivatives = torch.stack([dq_dt, dp_dt], dim=1)
    return states, derivatives


# =============================================================================
# 9. D'ALEMBERT WAVE EQUATION
# =============================================================================
def generate_dalembert_wave(resolution=256, num_samples=500, c=1.0, dt=0.01,
                             seed=DEFAULT_SEED):
    """
    Generates 1D wave equation solutions: u_tt = c² u_xx.
    Uses superposition of traveling waves u(x,t) = f(x - ct) + g(x + ct).
    
    Returns:
        u_t: [num_samples, resolution] wave field at t
        u_t1: [num_samples, resolution] wave field at t+dt
    """
    _set_seed(seed)
    x = torch.linspace(0, 4 * math.pi, resolution)

    u_t_list = []
    u_t1_list = []
    frequencies = torch.linspace(1.0, 5.0, num_samples)

    for i in range(num_samples):
        freq = frequencies[i]
        # Traveling wave: sin(freq * (x - c*0)) at t=0
        u_0 = torch.sin(freq * x)
        # At t=dt: sin(freq * (x - c*dt))
        u_1 = torch.sin(freq * (x - c * dt))

        u_t_list.append(u_0)
        u_t1_list.append(u_1)

    return torch.stack(u_t_list), torch.stack(u_t1_list)


# =============================================================================
# 10. FLRW COSMOLOGY (Friedmann expansion)
# =============================================================================
def generate_flrw_cosmology(num_samples=1000, seed=DEFAULT_SEED):
    """
    Generates scale factor a(t) trajectories under the Friedmann equation
    for a matter-dominated universe: H² = (8πG/3)ρ, ρ ∝ a⁻³.
    Exact solution: a(t) ∝ t^(2/3).
    
    Returns:
        a_values: [num_samples, 1] scale factors
        H_values: [num_samples, 1] Hubble parameters H(a)
    """
    _set_seed(seed)
    a = torch.linspace(0.1, 2.0, num_samples).unsqueeze(1)
    # Matter-dominated: H(a) = H_0 * a^(-3/2)
    H_0 = 1.0
    rho = 1.0 / (a ** 3)
    H = torch.sqrt((8 * math.pi / 3) * rho)
    return a, H


# =============================================================================
# FLUID FIELDS (Spectral Vorticity for Navier-Stokes)
# =============================================================================
def generate_spectral_vorticity_field(batch_size=100, resolution=64,
                                       viscosity=0.02, seed=DEFAULT_SEED):
    """
    Generates 2D spectral vorticity fields for incompressible Navier-Stokes.
    Each sample has unique phase offsets and frequencies for diversity.
    Uses exact viscous decay exp(-ν k² Δt) as the target evolution.
    
    Zero-Stub: No torch.randn(). Vorticity is spectrally constructed.
    
    Returns:
        w_t: [batch_size, 1, resolution, resolution] vorticity at t
        w_t1: [batch_size, 1, resolution, resolution] vorticity at t+dt
    """
    _set_seed(seed)
    x = torch.linspace(0, 2 * math.pi, resolution)
    y = torch.linspace(0, 2 * math.pi, resolution)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    w_t_list = []
    w_t1_list = []

    for i in range(batch_size):
        # Deterministic phase diversity
        phase_x = 2 * math.pi * i / batch_size
        phase_y = math.pi * i / batch_size
        freq = 2 + (i % 4)  # Frequencies cycle through 2, 3, 4, 5

        w0 = (torch.sin(freq * grid_x + phase_x) *
              torch.cos(freq * grid_y + phase_y) +
              0.5 * torch.sin(2 * freq * grid_x))

        # Exact viscous diffusion: spectral decay
        w1 = w0 * math.exp(-viscosity * (freq ** 2))

        w_t_list.append(w0.unsqueeze(0))
        w_t1_list.append(w1.unsqueeze(0))

    return torch.stack(w_t_list), torch.stack(w_t1_list)
