"""Camera calibration overlay: live feed blended over reference PNG.

Launch via calibrate_camera.sh (same Hydra / task pattern as valid_space.sh).

Camera-only: does NOT construct TienYiEnv, does NOT enable arms / switch
controllers / send joint or pose commands. Only loads cameras from
configuration.toml and overlays them on online_image_{key}.png.
"""

from __future__ import annotations

import copy
import sys
import traceback
from pathlib import Path

try:
    import cv2
    import hydra
    import numpy as np
    from xrocs.core.config_loader import ConfigLoader
    from xrocs.core.station_loader import StationLoader
    from rl_envs.utils.decode_image import decode_image
except Exception as e:
    print(f"Error in import: {e}")
    traceback.print_exc()
    sys.exit(1)

print("import success")

CONFIG_TOML = "/home/ubuntu/Documents/configuration.toml"

# Live-camera alpha in [0, 1]; reference uses (1 - alpha).
DEFAULT_ALPHA = 0.45
ALPHA_STEP = 0.05


def _load_reference_images(exp_dir: Path, image_keys: list[str]) -> dict[str, np.ndarray]:
    """Load BGR reference PNGs named online_image_{key}.png from exp_dir."""
    refs: dict[str, np.ndarray] = {}
    missing = []
    for key in image_keys:
        # path = exp_dir / f"online_image_{key}.png"
        path = exp_dir / f"frame_0_camera_{key}.png"
        if not path.is_file():
            missing.append(str(path))
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            missing.append(str(path))
            continue
        refs[key] = img
        print(f"loaded reference: {path}  shape={img.shape}")
    if missing:
        raise FileNotFoundError(
            "Missing / unreadable reference PNGs for image_keys "
            f"{image_keys}:\n  " + "\n  ".join(missing)
        )
    return refs


def _make_camera_only_station(image_keys: list[str]):
    """Build a station with cameras only — robot / hand / chassis / FTS disabled."""
    cfg = copy.deepcopy(ConfigLoader(CONFIG_TOML).get_config())

    # Never instantiate or connect arms / grippers / chassis / force sensors.
    if "robot" in cfg:
        cfg["robot"]["enable"] = False
    if "hand" in cfg:
        for hand_cfg in cfg["hand"].values():
            if isinstance(hand_cfg, dict):
                hand_cfg["enable"] = False
    if "chassis" in cfg and isinstance(cfg["chassis"], dict):
        cfg["chassis"]["enable"] = False
    if "force_sensor" in cfg:
        fs = cfg["force_sensor"]
        if isinstance(fs, dict) and "enable" in fs and not any(
            isinstance(v, dict) for v in fs.values()
        ):
            fs["enable"] = False
        else:
            for sensor_cfg in fs.values() if isinstance(fs, dict) else []:
                if isinstance(sensor_cfg, dict):
                    sensor_cfg["enable"] = False

    # Keep only the cameras listed in robot_config.image_keys.
    cam_cfg = cfg.get("camera", {})
    for name, one in list(cam_cfg.items()):
        if not isinstance(one, dict):
            continue
        one["enable"] = name in image_keys

    station = StationLoader(cfg).generate_station_handle()
    # connect() only iterates robot/hand dicts; both are empty here → no-op.
    station.connect()
    print(
        "camera-only station ready; cameras:",
        list(station.get_camera_handle().keys()),
        "| robots:",
        list(station.get_robot_handle().keys()),
        "| hands:",
        list(station.get_gripper_handle().keys()),
    )
    return station


def _fetch_live_images_bgr(station, image_keys: list[str], bgr2rgb: bool) -> dict[str, np.ndarray]:
    """Read cameras only via get_camera_state() — never get_robot_state / execute_action."""
    images_raw, _ = station.get_camera_state()
    out: dict[str, np.ndarray] = {}
    for key in image_keys:
        if key not in images_raw or images_raw[key] is None:
            continue
        # Same decode path as env obs; display uses BGR so convert back if needed.
        rgb, _ = decode_image(images_raw[key], None, bgr2rgb=bgr2rgb)
        if rgb is None:
            continue
        out[key] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if bgr2rgb else rgb
    return out


def _match_size(live_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    h, w = ref_bgr.shape[:2]
    if live_bgr.shape[0] == h and live_bgr.shape[1] == w:
        return live_bgr
    return cv2.resize(live_bgr, (w, h), interpolation=cv2.INTER_LINEAR)


def _blend(ref_bgr: np.ndarray, live_bgr: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    live = _match_size(live_bgr, ref_bgr)
    return cv2.addWeighted(live, alpha, ref_bgr, 1.0 - alpha, 0.0)


def _compose_panel(ref_bgr: np.ndarray, live_bgr: np.ndarray, alpha: float) -> np.ndarray:
    """Single view: reference as bottom layer, live on top with opacity ``alpha``."""
    return _blend(ref_bgr, live_bgr, alpha)


def _annotate(frame: np.ndarray, key: str, alpha: float) -> np.ndarray:
    out = frame.copy()
    cv2.putText(
        out,
        f"{key}  live(alpha={alpha:.2f}) over ref   [ / ] alpha   q quit",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out

def run_calibration(env_cfg) -> None:
    image_keys = list(env_cfg.robot_config.image_keys)
    bgr2rgb = bool(getattr(env_cfg.robot_config, "bgr2rgb", True))
    # exp_dir = Path.cwd().resolve()
    exp_dir = Path("/home/ubuntu/Dev/hermine/HIL-RL-Project/HIL-RL/experiments/install_handle/")
    print(f"experiment dir: {exp_dir}")
    print(f"image_keys: {image_keys}")

    refs = _load_reference_images(exp_dir, image_keys)
    station = _make_camera_only_station(image_keys)

    alpha = DEFAULT_ALPHA
    for name in (f"calib_{key}" for key in image_keys):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    print(
        "Calibration overlay running (cameras only — arms never enabled).\n"
        "  Live camera blended over reference PNG (adjustable alpha).\n"
        "  Keys: '[' decrease live alpha, ']' increase, 'q' quit."
    )

    try:
        while True:
            live_bgr = _fetch_live_images_bgr(station, image_keys, bgr2rgb=bgr2rgb)
            for key in image_keys:
                if key not in live_bgr:
                    print(f"warning: live image missing key={key}, available={list(live_bgr)}")
                    continue
                panel = _compose_panel(refs[key], live_bgr[key], alpha)
                cv2.imshow(f"calib_{key}", _annotate(panel, key, alpha))

            key_code = cv2.waitKey(1) & 0xFF
            if key_code in (ord("q"), 27):
                break
            if key_code == ord("["):
                alpha = max(0.0, alpha - ALPHA_STEP)
            elif key_code == ord("]"):
                alpha = min(1.0, alpha + ALPHA_STEP)
    finally:
        cv2.destroyAllWindows()
        try:
            station.close()
        except Exception:
            pass


@hydra.main(config_path="./cfg", config_name="config", version_base=None)
def main(env_cfg):
    print("before run_calibration ...")
    run_calibration(env_cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"In calibrate_camera.py: [{type(e).__name__}] {e!r}")
        traceback.print_exc()
        sys.exit(1)
