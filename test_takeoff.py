import airsim
import time

print("Connecting...")
client = airsim.MultirotorClient()
client.confirmConnection()

print("Resetting...")
client.reset()
client.enableApiControl(True)
client.armDisarm(True)

time.sleep(1)

print("Taking off...")
client.takeoffAsync().join()

p = client.getMultirotorState().kinematics_estimated.position
print(f"Altitude after takeoff: {-p.z_val}")

print("Moving up...")
client.moveByVelocityAsync(0, 0, -3.0, 2.0).join()

p = client.getMultirotorState().kinematics_estimated.position
print(f"Altitude after move: {-p.z_val}")

print("Hovering...")
client.hoverAsync().join()
time.sleep(2)

p = client.getMultirotorState().kinematics_estimated.position
print(f"Altitude after hover: {-p.z_val}")

client.armDisarm(False)
client.enableApiControl(False)
print("Done.")
