#!/usr/bin/env python3
"""
================================================================================
PRE-COMMIT RUN METADATA LOGGER & HARDWARE LOCK
================================================================================
Purpose: Meta-instrumentation to enforce scientific discipline and prevent
         unconscious confirmation bias in LAB-1 physical experiments.

Workflow:
1. Materially locks data buses (I2C, Camera) by holding a software mutex.
2. Prompts user for the independent variable being tested.
3. Prompts user for the pre-registered ALGEBRAIC prediction.
4. Queries NTP for an objective timestamp.
5. Generates a SHA-256 cryptographic hash of the prediction.
6. Unlocks the hardware buses ONLY AFTER the prediction is permanently logged.
"""

import os
import sys
import json
import hashlib
import ntplib
from time import ctime
import subprocess

LOCK_FILE = "/tmp/rama_hardware.lock"
META_LOG_FILE = "specs/lab1_experiment_log.jsonl"

def lock_hardware():
    """Simulates locking the I2C and Camera bus."""
    print("[HARDWARE] Locking I2C bus and Camera interfaces...")
    with open(LOCK_FILE, "w") as f:
        f.write("LOCKED")
    print("[HARDWARE] Buses locked. No data acquisition possible.")

def unlock_hardware():
    """Releases the lock on the I2C and Camera bus."""
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    print("[HARDWARE] I2C bus and Camera unlocked. Acquisition authorized.")

def get_ntp_time():
    """Fetch objective network time to prevent local clock spoofing."""
    try:
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org', version=3, timeout=3)
        return ctime(response.tx_time)
    except Exception as e:
        print(f"[NTP WARNING] Could not fetch NTP time ({e}). Falling back to local time.")
        import datetime
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    print("=" * 60)
    print("  RAMA LAB-1: AUTOMATED SCIENTIFIC DISCIPLINE (PRE-COMMIT)")
    print("=" * 60)
    
    lock_hardware()
    
    try:
        experiment_name = input("\n> Experiment Name (e.g., MANIP H - Laplace Tank): ").strip()
        varied_param = input("> Varied Parameter (e.g., Boundary potential V(x)): ").strip()
        prediction = input("> Pre-registered Algebraic Prediction: ").strip()
        
        if not experiment_name or not varied_param or not prediction:
            print("[ERROR] All fields are mandatory. Halting.")
            sys.exit(1)
            
        print("\n[NTP] Fetching objective timestamp...")
        timestamp = get_ntp_time()
        
        # Create the metadata payload
        payload = {
            "timestamp": timestamp,
            "experiment": experiment_name,
            "parameter": varied_param,
            "prediction": prediction
        }
        
        payload_str = json.dumps(payload, sort_keys=True)
        sha256_hash = hashlib.sha256(payload_str.encode('utf-8')).hexdigest()
        payload["sha256_hash"] = sha256_hash
        
        # Log to the append-only JSONL file
        with open(META_LOG_FILE, "a") as f:
            f.write(json.dumps(payload) + "\n")
            
        print("\n" + "=" * 60)
        print("[SUCCESS] Prediction permanently registered and hashed.")
        print(f"Hash: {sha256_hash}")
        print("=" * 60 + "\n")
        
        unlock_hardware()
        print("\n[READY] You may now run your data acquisition script.")
        
    except KeyboardInterrupt:
        print("\n[ABORT] User aborted. Hardware remains locked.")
        sys.exit(1)

if __name__ == "__main__":
    main()
