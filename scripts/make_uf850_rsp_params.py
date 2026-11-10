from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import yaml

urdf = Path(sys.argv[1]).read_text(encoding="utf-8")

assert ET.fromstring(urdf).tag == "robot"

params = {
    "robot_state_publisher": {
        "ros__parameters": {
            "use_sim_time": True,
            "robot_description": urdf,
        }
    }
}

Path(sys.argv[2]).write_text(
    yaml.safe_dump(params, sort_keys=False)
)

print("Generated robot_state_publisher parameters")
