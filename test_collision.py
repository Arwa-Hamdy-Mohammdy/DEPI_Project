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
info1 = client.simGetCollisionInfo()
print(f"Collision after takeoff: {info1.has_collided} with {info1.object_name}")

print("Moving up...")
client.moveByVelocityAsync(0, 0, -3.0, 2.0).join()
info2 = client.simGetCollisionInfo()
print(f"Collision after move: {info2.has_collided} with {info2.object_name}")

p = client.getMultirotorState().kinematics_estimated.position
print(f"Altitude after move: {-p.z_val}")

client.armDisarm(False)
client.enableApiControl(False)
print("Done.")
