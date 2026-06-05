import asyncio
from mavsdk import System
from mavsdk.mission import MissionItem, MissionPlan


async def upload_and_execute_mission(mission_items: list[MissionItem], connection_url: str = "udpin://0.0.0.0:14540") -> None:
    """
    Connects to PX4, verifies EKF health, uploads the mission, arms, executes, and monitors progress.
    """
    drone = System()
    print(f"Connecting to PX4 SITL on {connection_url}...")
    await drone.connect(system_address=connection_url)

    # 1. Wait for connection
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Connected to drone!")
            break

    # 2. Wait for Global Position Estimate (EKF readiness) with timeout
    print("Waiting for EKF global position estimate...")
    max_wait = 60
    elapsed = 0
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("✅ Global position estimate OK.")
            break
        if elapsed >= max_wait:
            raise RuntimeError("EKF health check timed out after 60s.")
        await asyncio.sleep(1)
        elapsed += 1

    # 3. Clear and Upload
    print("Clearing existing mission buffer...")
    await drone.mission.clear_mission()

    print(f"Uploading {len(mission_items)} mission items...")
    mission_plan = MissionPlan(mission_items)
    await drone.mission.upload_mission(mission_plan)
    print("✅ Mission uploaded successfully.")

    # 4. Arm and Execute
    print("Arming the drone...")
    try:
        await drone.action.arm()
    except Exception as exc:
        raise RuntimeError(
            "Arm denied — fix compass/health in PX4 (pxh> param set CAL_MAG0_ID 197388; "
            "CAL_MAG1_ID 197644; CAL_MAG0_PRIO 50), restart SITL, rerun sitl_failsafe_override.py"
        ) from exc

    print("Starting mission (Auto-Takeoff)...")
    await drone.mission.start_mission()
    print("🚀 Mission is now running!")

    # 5. Monitor Progress to keep event loop alive
    print("Monitoring mission progress...")
    async for progress in drone.mission.mission_progress():
        print(f"Waypoint {progress.current} / {progress.total}")
        if progress.current == progress.total:
            print("✅ Mission complete.")
            break
