import os
import json
import subprocess
import time

# 1. Define paths
default_airsim_path = r"C:\Users\simoo\OneDrive\Documents\AirSim\settings.json"
unreal_exe_path = r"C:\cityenviron\CityEnviron\WindowsNoEditor\CityEnviron.exe"

# 2. Define the core, optimized settings
base_settings = {
    "SettingsVersion": 1.2,
    "SimMode": "Multirotor",
    "ViewMode": "NoDisplay",
    "EngineSound": False,
    "CameraDefaults": {
        "CaptureSettings": [
            {
                "ImageType": 2,
                "Width": 64,
                "Height": 64
            },
            {
                "ImageType": 5,
                "Width": 64,
                "Height": 64
            }
        ]
    }
}

# The single port we want to use to save RAM/VRAM
ports = [41451,41452]
launched_processes = []

print("Commencing dynamic AirSim launch sequence (Single Port Fallback)...")

os.makedirs(os.path.dirname(default_airsim_path), exist_ok=True)

for port in ports:
    print(f"--> Configuring settings for Port {port}...")
    base_settings["LocalHostPort"] = port
    
    with open(default_airsim_path, 'w') as f:
        json.dump(base_settings, f, indent=4)
    
    print(f"--> Launching Unreal Engine on Port {port}...")
    process = subprocess.Popen(
        [unreal_exe_path, "-ResX=640", "-ResY=480", "-WINDOWED"]
    )
    launched_processes.append(process)
    
    # Wait to ensure the game reads the file
    time.sleep(10.0)

print("\nEnvironment successfully launched!")
print("Leave this script running. Press Ctrl+C to kill the AirSim instance.")

try:
    for p in launched_processes:
        p.wait()
except KeyboardInterrupt:
    print("\nShutting down AirSim instance...")
    for p in launched_processes:
        p.terminate()