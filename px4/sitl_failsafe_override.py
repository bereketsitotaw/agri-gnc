"""
Configure PX4 SITL for robust autonomous mission testing via MAVSDK.

Run after starting jMAVSim/Gazebo and before upload_and_execute_mission or
run_sitl_mission.py — especially when changing global home (e.g. Hawassa).

Note: jMAVSim does not provide GPS yaw. Keep SYS_HAS_MAG=1 and EKF2_MAG_TYPE=0
(automatic mag fusion). Forcing GPS heading (EKF2_MAG_TYPE=1) causes permanent
"no heading reference" in SITL.

Usage:
    PYTHONPATH=. python3 px4/sitl_failsafe_override.py

    Then manually restart SITL (Ctrl+C, relaunch make px4_sitl ...). PX4 SITL
    denies MAVSDK reboot commands; params are persisted to parameters.bson.
"""

import asyncio

from mavsdk import System
from mavsdk.param import ParamError

CONNECTION_URL = "udpin://0.0.0.0:14540"

# (param_name, value, human-readable description)
SITL_OVERRIDES: list[tuple[str, int, str]] = [
    # Magnetometer / EKF — Gazebo has no GPS yaw; use virtual compass at new home
    ("SYS_HAS_MAG", 1, "Enable magnetometer (required for SITL heading)"),
    ("EKF2_MAG_TYPE", 0, "Automatic mag fusion (do not force GPS heading)"),
    ("COM_ARM_MAG_STR", 0, "Relax magnetometer strength arming check"),
    ("COM_ARM_MAG_ANG", 180, "Disable magnetometer angle arming check"),
    # Gazebo SITL sim magnetometer IDs (see ROMFS px4fmu_common/init.d-posix/rcS)
    ("CAL_MAG0_ID", 197388, "Sim mag 0 device ID (gz magnetometer_sensor)"),
    ("CAL_MAG0_PRIO", 50, "Primary magnetometer priority"),
    ("CAL_MAG1_ID", 197644, "Sim mag 1 device ID"),
    # Power — ignore missing simulated battery
    ("CBRK_SUPPLY_CHK", 894281, "Bypass supply/battery health circuit breaker"),
    # Link loss — do not RTL on simulated RC/datalink dropouts
    ("NAV_RCL_ACT", 0, "RC loss action: disabled"),
    ("NAV_DLL_ACT", 0, "Datalink loss action: disabled"),
    # USB preflight
    ("CBRK_USB_CHK", 197848, "Bypass USB connection preflight check"),
]


async def apply_sitl_overrides() -> None:
    drone = System()
    print(f"Connecting to PX4 SITL on {CONNECTION_URL}...")
    await drone.connect(system_address=CONNECTION_URL)

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Connected to drone.")
            break

    print("\nApplying SITL robustness parameters...")
    failed: list[str] = []

    for name, value, description in SITL_OVERRIDES:
        try:
            await drone.param.set_param_int(name, value)
            print(f"  ✅ {name} = {value}  ({description})")
        except ParamError as exc:
            print(f"  ❌ {name} failed: {exc}")
            failed.append(name)
        except Exception as exc:
            print(f"  ❌ {name} unexpected error: {exc}")
            failed.append(name)

    if failed:
        print(f"\n⚠️  {len(failed)} parameter(s) could not be set: {', '.join(failed)}")
    else:
        print("\n✅ All parameters set successfully.")

    # PX4 SITL denies software reboot via MAVSDK (COMMAND_DENIED). Params are
    # saved to parameters.bson — restart the simulator manually to apply them.
    # await drone.action.reboot()

    print("\n✅ Done. Restart SITL manually (Ctrl+C, then relaunch make px4_sitl).")
    print("   Wait for 'INFO [commander] Ready for takeoff!' before running missions.")
    print("   Run this script BEFORE run_sitl_mission.py — not during flight.")


if __name__ == "__main__":
    asyncio.run(apply_sitl_overrides())
