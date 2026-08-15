"""
TNN Shared Numerical Integrators
=================================
Physics-preserving time-stepping methods for Hamiltonian systems.
Replaces ad-hoc Euler integration across the codebase.

Policy: All rollout evaluations MUST use RK4 or Störmer-Verlet.
        Plain Euler is banned per Scientific_Rigor_Audit.md §1.
"""
import torch


def rk4_step(derivative_fn, q, p, masses, dt):
    """
    Runge-Kutta 4th-order integrator for Hamiltonian systems.
    
    Args:
        derivative_fn: Callable(q, p, masses) -> (dq/dt, dp/dt)
        q: Position tensor [batch, N, D]
        p: Momentum tensor [batch, N, D]
        masses: Mass tensor [N] or [N, 1]
        dt: Time step (scalar)
    
    Returns:
        q_next, p_next: Advanced state tensors
    """
    k1_q, k1_p = derivative_fn(q, p, masses)

    q_k2 = q + 0.5 * dt * k1_q
    p_k2 = p + 0.5 * dt * k1_p
    k2_q, k2_p = derivative_fn(q_k2, p_k2, masses)

    q_k3 = q + 0.5 * dt * k2_q
    p_k3 = p + 0.5 * dt * k2_p
    k3_q, k3_p = derivative_fn(q_k3, p_k3, masses)

    q_k4 = q + dt * k3_q
    p_k4 = p + dt * k3_p
    k4_q, k4_p = derivative_fn(q_k4, p_k4, masses)

    q_next = q + (dt / 6.0) * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    p_next = p + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)

    return q_next, p_next


def rk4_step_flat(derivative_fn, x, dt):
    """
    RK4 for flat state vectors x = (q, p) concatenated.
    
    Args:
        derivative_fn: Callable(x) -> dx/dt
        x: State tensor [batch, state_dim] (requires_grad may be True)
        dt: Time step (scalar)
    
    Returns:
        x_next: Advanced state tensor
    """
    k1 = derivative_fn(x)
    k2 = derivative_fn(x + 0.5 * dt * k1)
    k3 = derivative_fn(x + 0.5 * dt * k2)
    k4 = derivative_fn(x + dt * k3)

    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next


def stormer_verlet_step(force_fn, q, p, masses, dt):
    """
    Störmer-Verlet (Leapfrog) symplectic integrator.
    Exactly conserves phase-space volume (Liouville's theorem).
    
    Args:
        force_fn: Callable(q) -> force tensor (= -dV/dq)
        q: Position tensor [batch, N, D]
        p: Momentum tensor [batch, N, D]
        masses: Mass tensor broadcastable to p
        dt: Time step (scalar)
    
    Returns:
        q_next, p_next: Advanced state tensors
    """
    # Half-step momentum
    f = force_fn(q)
    p_half = p + 0.5 * dt * f

    # Full-step position
    q_next = q + dt * p_half / masses

    # Half-step momentum (with updated position)
    f_next = force_fn(q_next)
    p_next = p_half + 0.5 * dt * f_next

    return q_next, p_next
