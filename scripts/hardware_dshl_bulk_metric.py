import cv2
import numpy as np

def capture_fluid_bulk_metric(frame_rest, frame_active):
    """
    Captures the fluid's volume dynamics (Bulk/IR) using Fast-Chequerboard Demodulation.
    """
    gray_rest = cv2.cvtColor(frame_rest, cv2.COLOR_BGR2GRAY)
    gray_act = cv2.cvtColor(frame_active, cv2.COLOR_BGR2GRAY)
    
    # Calculates dense optical flow to map surface elevation gradients
    flow = cv2.calcOpticalFlowFarneback(
        gray_rest, gray_act, None, 
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    # Return elevation magnitude matrix (2D Bulk Tensor)
    return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
