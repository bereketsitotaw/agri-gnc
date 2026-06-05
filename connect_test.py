import asyncio
from mavsdk import System

async def test_connection():
    drone = System()
    print("Connecting to PX4 SITL on udpin://0.0.0.0:14540...")
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ MAVSDK Connected to drone!")
            break

    print("Waiting for EKF global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("✅ EKF is healthy. Drone knows where it is.")
            break
        await asyncio.sleep(1)

    print("Attempting to arm...")
    await drone.action.arm()
    print("✅ Armed successfully!")
    
    await asyncio.sleep(2)
    
    print("Attempting to disarm...")
    await drone.action.disarm()
    print("✅ Disarmed successfully! The UDP pipeline is functional.")

if __name__ == "__main__":
    asyncio.run(test_connection())
