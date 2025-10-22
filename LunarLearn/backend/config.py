import os
import yaml
import argparse
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(os.getenv("LUNAR_CONFIG", "lunar_config.yaml"))

def load_yaml_config(path: str = None) -> dict:
    """Load configuration from a YAML file."""
    cfg_path = Path(path or os.getenv("LUNAR_CONFIG", DEFAULT_CONFIG_PATH))
    if not cfg_path.exists():
        print(f"[LunarLearn] No config found at {cfg_path}. Using defaults.")
        return {}
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}

def parse_cli_args() -> dict:
    """Parse CLI overrides (used for runtime config tweaking)."""
    parser = argparse.ArgumentParser(description="LunarLearn Config Override")

    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], help="Device to use")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32", "float64"], help="Floating point precision")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--mixed_precision", type=str, choices=["true", "false"], help="Enable mixed precision")
    parser.add_argument("--safe_factor", type=float, help="Memory safety factor")
    parser.add_argument("--scaling_factor", type=int, help="Loss scaling factor")
    parser.add_argument("--autograd", type=str, choices=["true", "false"], help="Enable autograd")

    args, _ = parser.parse_known_args()

    cli_config = {}
    for key, value in vars(args).items():
        if value is not None:
            # Cast booleans properly
            if value == "true":
                value = True
            elif value == "false":
                value = False
            cli_config[key] = value

    return cli_config

def merge_configs(base: dict, override: dict) -> dict:
    """Merge CLI overrides into YAML base config (CLI wins)."""
    final = base.copy()
    final.update(override)
    return final

def load_config() -> dict:
    """Main config loader: YAML + CLI overrides."""
    cli = parse_cli_args()
    yaml_cfg = load_yaml_config(cli.get("config"))
    return merge_configs(yaml_cfg, cli)

# === The global CONFIG dict you import elsewhere ===
CONFIG = load_config()