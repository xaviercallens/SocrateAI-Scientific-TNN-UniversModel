IMPLEMENTATION SPECIFICATION: DSHL (LOW-TIER)1. Global System ArchitectureTo manage the discrepancy between high-level computing (Tensor Neural Networks, Video Processing) and low-level deterministic execution (Stepper motors, kHz acoustic sampling), the architecture is split into two layers:The Compute & Control Layer (The Brain): A Raspberry Pi 4 (4GB+ RAM) or a standard Linux/Windows Laptop. It hosts the Control Center, runs the Digital Twin simulations, processes Fast-Chequerboard Demodulation (FCD) via OpenCV, and executes the PyTorch TNN.The Real-Time DAQ Layer (The Spinal Cord): A Raspberry Pi Pico (RP2040 micro-controller, ~$5). It handles microsecond-accurate hardware polling and PWM generation without Operating System-level interruptions.Unified Low-Tier Bill of Materials (BOM)SubsystemComponentsEst. CostOpto-Mechanics (Labs 1-2)2x Salvaged DVD Stepper mechanisms, 2x A4988 Stepper Drivers, 1x BPW34 Photodiode, 1x ADS1115 I2C ADC, 5mW Laser module.~$15Hydrodynamics (Lab 3)60L clear plastic storage tote, 12V DC Aquarium Pump, 5V Solenoid (for UV perturbations), IRLZ44N MOSFETs, USB Webcam.~$35Acoustic Array (Lab 4)16x Piezoelectric discs (salvaged buzzers), 1x CD74HC4067 (16-channel analog multiplexer).~$10Microcontroller & PowerRaspberry Pi Pico, 12V/5A Power Supply, Breadboards, Jumper wires.~$20Total< $802. Electronic Plan Generation & SchematicsThe hardware acts as a modular shield connected to the Pi Pico. Logic levels (3.3V) are strictly isolated from actuator power (5V-12V) using a common ground.Module A: Opto-Mechanical Scanners (BOMA-2D)X/Y Axes: STEP and DIR pins of the A4988 drivers connect to Pico GPIOs (e.g., GP14-GP17). Power is supplied externally (12V to VMOT) with a common ground.Module B: Acoustic Tomography Array (ATA)CD74HC4067 Multiplexer: The SIG (signal) pin connects to the Pico's ADC0 (GP26). The address control pins S0-S3 connect to Pico GP2-GP5. The 16 Piezo sensors connect to the multiplexer channels C0-C15.Module C: Hydrodynamic Actuation & Metric GenerationPump Control: Pico GP20 (configured as PWM) connects to the Gate of an IRLZ44N MOSFET (via a 330$\Omega$ resistor) to regulate the 12V pump speed (defining the Froude number). A 1N4007 flyback diode must be placed across the pump terminals.Perturber: Pico GP21 triggers a 5V solenoid via a TIP120 transistor to inject discrete "plucks" (UV waves) into the water.3. Drivers & Metric Capture PipelinesA. High-Frequency Acoustic Driver (MicroPython on Pico)The Pico runs a tight, deterministic loop to sweep the 16 piezo sensors on the tank's boundary at high speed, streaming the 1D tensor to the Control Center via USB Serial.Pythonimport machine, sys, utime

adc = machine.ADC(26)
s_pins = [machine.Pin(i, machine.Pin.OUT) for i in range(2, 6)]

def stream_boundary_tensor():
    while True:
        tensor = []
        for i in range(16):
            for j in range(4): # Set MUX 4-bit binary address
                s_pins[j].value((i >> j) & 1)
            utime.sleep_us(10) # Settling time for multiplexer
            tensor.append(adc.read_u16())
        # Stream comma-separated tensor to the Host computer
        sys.stdout.write(','.join(map(str, tensor)) + '\n')
        utime.sleep_ms(2) # Yield for ~500Hz full-array sampling
B. Visual Bulk Metric Driver (Python on Host)Captures the fluid's volume dynamics (Bulk/IR) using Fast-Chequerboard Demodulation.Pythonimport cv2
import numpy as np

def capture_fluid_bulk_metric(frame_rest, frame_active):
    gray_rest = cv2.cvtColor(frame_rest, cv2.COLOR_BGR2GRAY)
    gray_act = cv2.cvtColor(frame_active, cv2.COLOR_BGR2GRAY)
    
    # Calculates dense optical flow to map surface elevation gradients
    flow = cv2.calcOpticalFlowFarneback(
        gray_rest, gray_act, None, 
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    # Return elevation magnitude matrix (2D Bulk Tensor)
    return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2) 
4. The Control Center & The Epistemological LockThe Control Center (a Python script running on the Host) strictly enforces the Lab-0 Discipline. Hardware actuation is locked until a theoretical prediction is generated and hashed.Pythonimport hashlib, json, time

def epistemological_lock(simulation_config, predicted_tensor):
    # 1. Hash the Digital Twin's prediction BEFORE hardware actuation
    meta = {
        "config": simulation_config, 
        "prediction": predicted_tensor.tolist(), 
        "timestamp": time.time()
    }
    run_hash = hashlib.sha256(json.dumps(meta).encode()).hexdigest()
    
    with open(f"LOCKED_RUN_{run_hash[:8]}.meta", "w") as f:
        json.dump(meta, f)
        
    print(f"[LOCK ENGAGED] Hash: {run_hash[:8]}. Hardware unlocked.")
    return run_hash

# Execution Flow:
# lock_hash = epistemological_lock(config, digital_twin_output)
# serial.write(b"ACTIVATE_PUMP:1.2_FROUDE")
# bulk_tensor, boundary_tensor = capture_metrics()
# calculate_residuals(digital_twin_output, bulk_tensor)
5. Guidelines: The "Google Antigravity" Agentic FrameworkTo manage the complexity of this laboratory without treating Artificial Intelligence as an infallible "black box," we define the Google Antigravity Framework. This is a Multi-Agent System (MAS) deployed via Google Vertex AI Agent Builder or Gemini 1.5 Pro APIs."Antigravity" acts as a swarm of AI lab assistants that lift the coding and analytical burden, but are tightly constrained by system prompts to enforce philosophical hygiene.Agent 1: The Theoretician (Digital Twin Generator)Role: Translates theoretical principles into deterministic simulation code.System Prompt / Guideline: "You are blinded to reality. Given experimental parameters (e.g., flow rate, tank depth), write the Python scipy/numpy scripts to solve the 1D Bernoulli equation and predict the exact Froude number. Output the theoretical coordinate of the analogue horizon ($r_h$). Your output will be cryptographically locked."Agent 2: The Engineer (Hardware & Routing)Role: Generates electronic plans, PCB netlists, and driver code.System Prompt / Guideline: "You act as the embedded systems developer. If the user provides a datasheet for a salvaged stepper motor or pump, generate the exact MicroPython code and GPIO pinout required to drive it safely using 3.3V logic and isolated 12V power."Agent 3: The TNN Architect (AI for Lab 4)Role: Constructs the Tensor Neural Network to map the Boundary to the Bulk.System Prompt / Guideline: "Write a PyTorch model mapping the 1D acoustic boundary tensor (ATA) to the 2D optical bulk tensor (FCD). You MUST enforce a strict linear Bottleneck dimension ($\chi$) to test the Area Law. Automatically generate a spatial Loss Heatmap to locate the 'Algorithmic Crash'—the geometrical coordinate where classical fluid physics fails and the P4 Dispersive Bounce emerges."Agent 4: The Epistemologist (The Honesty Clause Enforcer)Role: The most critical agent. Acts as an adversarial peer-reviewer based on the philosophy of Crowther, Linnemann, and Wüthrich.System Prompt / Guideline: "You are the Epistemological Auditor. You must hunt for 'petitio principii' (begging the question). Ensure the setup tests a syntactic mathematical isomorphism, NOT a fundamental ontology of astrophysical quantum gravity. Before approving any successful TNN mapping from Agent 3, you MUST mandate a Negative Control."Action Example (The No-Magic Theorem): If Agent 3 successfully reconstructs the bulk, the Epistemologist halts the pipeline and instructs the user: "Place an asymmetrical, chaotic obstacle in the tank to destroy the irrotational flow metric. Rerun Agent 3. If the TNN still succeeds, the AI is cheating by interpolating noise. If the TNN crashes, you have validated the structural integrity of the dual-scale holographic boundary."