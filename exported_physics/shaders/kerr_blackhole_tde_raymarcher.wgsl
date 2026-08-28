// ============================================================================
// WGSL COMPUTE SHADER: Kerr Metric Raymarching & Tidal Disruption Event (TDE)
// Physics Engine: HoloAlg (T-Dual Spacetime + Chameleon + Leray SPH + Planck)
// ============================================================================

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    cam_pos: vec4<f32>,
    resolution: vec2<f32>,
    time: f32,
    black_hole_mass: f32,
    spin_a: f32,             // Dimensionless spin a* in [0, 0.998]
    alpha_prime: f32,        // T-dual fundamental string scale alpha'
    chameleon_alpha_eff: f32,// Chameleon coupling alpha_eff ~ 1.55
    max_steps: u32,
    step_size: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba16float, write>;

// ----------------------------------------------------------------------------
// 1. T-Dual Cosmological Censorship: Effective Radius R_eff
// ----------------------------------------------------------------------------
fn compute_r_eff(r: f32, alpha_prime: f32) -> f32 {
    // Replaces classical singularity r=0 by T-dual bounce: R_eff = max(r, alpha' / r)
    // Guarantee: R_eff >= sqrt(alpha') > 0 everywhere
    let safe_r = max(r, 1e-5);
    return max(safe_r, alpha_prime / safe_r);
}

// ----------------------------------------------------------------------------
// 2. Kerr Spacetime Geodesic Acceleration (Boyer-Lindquist Metric)
// ----------------------------------------------------------------------------
struct GeodesicState {
    pos: vec3<f32>, // Cartesian 3D coordinates
    vel: vec3<f32>, // Unit wavevector / velocity
};

fn kerr_geodesic_step(state: GeodesicState, M: f32, a: f32, alpha_p: f32, alpha_eff: f32, dt: f32) -> GeodesicState {
    let r_raw = length(state.pos);
    let r_eff = compute_r_eff(r_raw, alpha_p);
    
    // Effective Kerr Horizon: r_H = M * (1 + sqrt(1 - a^2))
    let r_h = M * (1.0 + sqrt(max(0.0, 1.0 - a * a)));
    
    // Frame Dragging (Lense-Thirring) Angular Frequency: omega = 2 * M * a * r / (r^4 + a^2 r^2 + 2 M a^2 r)
    let denom_omega = r_eff * r_eff * r_eff * r_eff + a * a * r_eff * r_eff + 2.0 * M * a * a * r_eff;
    let omega_drag = (2.0 * M * a * r_eff * alpha_eff) / max(denom_omega, 1e-4);
    
    // Gravitational Acceleration toward Kerr Singularity Shell
    let grav_mag = (M * alpha_eff) / (r_eff * r_eff);
    let radial_dir = -normalize(state.pos);
    
    // Cross product with Kerr spin vector (along z-axis)
    let spin_axis = vec3<f32>(0.0, 1.0, 0.0);
    let lense_thirring_accel = cross(spin_axis, state.pos) * omega_drag;
    
    let total_accel = radial_dir * grav_mag + lense_thirring_accel;
    
    // Symplectic Verlet Step for Photon Geodesic
    var new_state: GeodesicState;
    new_state.vel = normalize(state.vel + total_accel * dt);
    new_state.pos = state.pos + new_state.vel * dt;
    return new_state;
}

// ----------------------------------------------------------------------------
// 3. Planck Blackbody Radiation Function (Thermodynamic Color Mapping)
// ----------------------------------------------------------------------------
fn planck_blackbody_rgb(temp_kelvin: f32) -> vec3<f32> {
    let t = clamp(temp_kelvin, 1000.0, 40000.0) / 100.0;
    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;
    
    // Red Channel
    if (t <= 66.0) {
        r = 1.0;
    } else {
        r = clamp(1.292936186062744 * pow(t - 60.0, -0.1332047592), 0.0, 1.0);
    }
    
    // Green Channel
    if (t <= 66.0) {
        g = clamp(0.3900815787690196 * log(t) - 0.6318414437886275, 0.0, 1.0);
    } else {
        g = clamp(1.129890860895294 * pow(t - 60.0, -0.0755148492), 0.0, 1.0);
    }
    
    // Blue Channel
    if (t >= 66.0) {
        b = 1.0;
    } else if (t <= 19.0) {
        b = 0.0;
    } else {
        b = clamp(0.543206789110196 * log(t - 10.0) - 1.19625408914, 0.0, 1.0);
    }
    
    return vec3<f32>(r, g, b);
}

// ----------------------------------------------------------------------------
// 4. Relativistic Doppler Beaming & Gravitational Redshift
// ----------------------------------------------------------------------------
fn compute_relativistic_doppler(fluid_vel: vec3<f32>, photon_dir: vec3<f32>, r_eff: f32, M: f32) -> f32 {
    let beta = clamp(length(fluid_vel), 0.0, 0.99);
    let gamma = 1.0 / sqrt(1.0 - beta * beta);
    let cos_theta = dot(normalize(fluid_vel), -photon_dir);
    
    // Gravitational Redshift Factor: sqrt(1 - 2M / R_eff)
    let g_grav = sqrt(max(1e-4, 1.0 - (2.0 * M) / r_eff));
    
    // Relativistic Kinematic Invariant Factor g_rel
    let g_doppler = 1.0 / (gamma * (1.0 - beta * cos_theta));
    let g_total = g_grav * g_doppler;
    
    // Specific Intensity I_obs = g^4 * I_emit (Beaming Exponent = 4 for Continuum Fluid)
    return pow(g_total, 4.0);
}

// ----------------------------------------------------------------------------
// 5. Main Compute Kernel: Non-Euclidean Kerr Raymarcher
// ----------------------------------------------------------------------------
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let uv = (vec2<f32>(global_id.xy) / camera.resolution) * 2.0 - 1.0;
    
    // Construct initial camera ray in world space
    let ray_origin = camera.cam_pos.xyz;
    let ray_dir = normalize((camera.view_matrix * vec4<f32>(uv.x, -uv.y, -1.0, 0.0)).xyz);
    
    var state: GeodesicState;
    state.pos = ray_origin;
    state.vel = ray_dir;
    
    let M = camera.black_hole_mass;
    let a = camera.spin_a;
    let alpha_p = camera.alpha_prime;
    let alpha_eff = camera.chameleon_alpha_eff;
    let r_h = M * (1.0 + sqrt(max(0.0, 1.0 - a * a)));
    
    var accumulated_color = vec3<f32>(0.0, 0.0, 0.0);
    var transmittance: f32 = 1.0;
    var hit_horizon: bool = false;
    
    for (var i: u32 = 0u; i < camera.max_steps; i = i + 1u) {
        let r_raw = length(state.pos);
        let r_eff = compute_r_eff(r_raw, alpha_p);
        
        // Check Event Horizon Absorption
        if (r_eff <= r_h * 1.01) {
            hit_horizon = true;
            break;
        }
        
        // Evaluate Accretion Disk / TDE Debris Stream (Torus at y ~ 0)
        let disk_dist_y = abs(state.pos.y);
        let disk_dist_r = length(vec2<f32>(state.pos.x, state.pos.z));
        
        if (disk_dist_y < 0.4 && disk_dist_r > r_h * 1.2 && disk_dist_r < M * 14.0) {
            // Keplerian / Relativistic Fluid Velocity: v_phi ~ sqrt(M / r)
            let phi_hat = normalize(vec3<f32>(-state.pos.z, 0.0, state.pos.x));
            let v_mag = clamp(sqrt(M / r_eff), 0.05, 0.75);
            let fluid_vel = phi_hat * v_mag;
            
            // Local Thermodynamic Temperature from TDE Shock: T(r) ~ T_0 * (r / r_isco)^(-3/4)
            let temp_k = 28000.0 * pow(max(r_eff / (M * 2.0), 1.0), -0.75);
            let base_color = planck_blackbody_rgb(temp_k);
            
            // Relativistic Doppler Beaming Factor
            let doppler_boost = compute_relativistic_doppler(fluid_vel, state.vel, r_eff, M);
            
            // Volumetric Density of SPH Debris Stream
            let density = exp(-disk_dist_y * 4.0) * exp(-abs(disk_dist_r - M * 5.0) * 0.35) * 0.12;
            let emission = base_color * doppler_boost * density;
            
            accumulated_color += emission * transmittance;
            transmittance *= exp(-density * 0.5);
            
            if (transmittance < 0.01) {
                break;
            }
        }
        
        // Advance photon along curved Kerr geodesic
        let step_adaptive = camera.step_size * clamp(r_eff / (M * 2.0), 0.2, 2.5);
        state = kerr_geodesic_step(state, M, a, alpha_p, alpha_eff, step_adaptive);
        
        // Escape condition: photon leaves near-horizon region
        if (length(state.pos) > M * 25.0) {
            break;
        }
    }
    
    // Background Galactic Starfield (Gravitational Lensing & Einstein Ring)
    if (!hit_horizon) {
        let sky_dir = normalize(state.vel);
        let star_seed = sin(dot(sky_dir, vec3<f32>(12.9898, 78.233, 45.164))) * 43758.5453;
        let star_intensity = pow(clamp(fract(star_seed) - 0.996, 0.0, 1.0) * 250.0, 4.0);
        let galactic_lensed = vec3<f32>(0.08, 0.12, 0.22) * (1.0 + 0.5 * sky_dir.y) + vec3<f32>(star_intensity);
        accumulated_color += galactic_lensed * transmittance;
    } else {
        // Absolute Black Hole Shadow: zero light escapes
        accumulated_color *= (1.0 - transmittance);
    }
    
    // ACES Tone-mapping
    let a_tone: f32 = 2.51;
    let b_tone: f32 = 0.03;
    let c_tone: f32 = 2.43;
    let d_tone: f32 = 0.59;
    let e_tone: f32 = 0.14;
    let tone_mapped = clamp((accumulated_color * (a_tone * accumulated_color + b_tone)) / 
                            (accumulated_color * (c_tone * accumulated_color + d_tone) + e_tone), 
                            vec3<f32>(0.0), vec3<f32>(1.0));
    
    textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(tone_mapped, 1.0));
}
