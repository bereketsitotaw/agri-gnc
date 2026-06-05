# agri-gnc

**UAV guidance, navigation, and control (GNC) for precision agricultural spraying.**

A four-layer Python pipeline that turns field polygons into PX4-ready spray missions: geometry → coverage path planning → wind-aware trajectory smoothing → MAVSDK upload. Validated in **PX4 SITL** (Gazebo) with automated graders for cross-track error (XTE), spray geofencing, and high-altitude thrust derating.

| | |
|---|---|
| **Stack** | Python 3.11+, PX4, MAVSDK, Shapely, pyproj |
| **Phase** | Phase I complete (SITL): drift crabbing, ρ(h) derating, multi-polygon geofencing |
| **License** | [MIT](LICENSE) |

---

## What it does

| Layer | Module | Role |
|-------|--------|------|
| L1 | `geometry/` | WGS84 polygons → UTM; spray geofence slicing |
| L2 | `cpp/` | Boustrophedon coverage path planning |
| L3 | `trajectory/` | Bézier turns, wind braking, drift model, energy-optimal speeds |
| L4 | `px4/` | MAVLink mission items + MAVSDK upload |
| — | `analysis/` | ULog XTE parsing, SITL graders, energy benchmarks |

---

## SITL results (representative)

| Metric | Result |
|--------|--------|
| Max XTE @ 5 m/s crosswind | **0.888 m** (≤ 0.97 m tolerance) |
| Geofence overspray @ 5.0 m | **0.0%** (dry-corridor mission) |
| Analytical energy vs. fixed 8 m/s | **−4.9%** on 515 m DEMO mission |
| Unit tests | `pytest tests/ -q` |

---

## Quick start

```bash
git clone https://github.com/bereketsitotaw/agri-gnc.git
cd agri-gnc
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.

pytest tests/ -q
python analysis/mission_energy_benchmark.py
```

### PX4 SITL

Requires a separate [PX4-Autopilot](https://github.com/PX4/PX4-Autopilot) checkout.

```bash
export PX4_HOME_LAT=7.050000 PX4_HOME_LON=38.470000 PX4_HOME_ALT=1708

# Terminal 1 (inside PX4-Autopilot):
make px4_sitl gz_x500

# Terminal 2 (this repo):
python px4/sitl_failsafe_override.py
ENABLE_SPRINT3_GEOFENCE=1 python run_sitl_mission.py
python analysis/ulog_parser.py
```

Key flags in `run_sitl_mission.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `USE_ENERGY_OPTIMAL` | `True` | Per-sweep optimal speed vs. fixed 8 m/s |
| `WIND_SPEED` | `5.0` | m/s |
| `WIND_FROM_DEG` | `180.0` | Meteorological wind FROM |
| `ENABLE_SPRINT3_GEOFENCE` | `1` | Multi-polygon spray ON/OFF geofencing |

---

## Repository layout

```
agri-gnc/
├── geometry/       # UTM, polygons, spray geofence
├── cpp/            # Boustrophedon CPP
├── trajectory/     # Arc smoother, drift, altitude physics, energy model
├── px4/            # MAVLink converter, mission uploader
├── analysis/       # ULog parser, graders, benchmarks
├── tests/
├── run_sitl_mission.py
└── requirements.txt
```

---

## Roadmap

- Phase II: HAL pump mapping (`MAV_CMD_DO_SET_SERVO` / GPIO)
- Phase V: field System ID, companion-computer deploy, DEM terrain following

---

## License

MIT — see [LICENSE](LICENSE).

**Author:** Bereket S. Kidane · 2026
