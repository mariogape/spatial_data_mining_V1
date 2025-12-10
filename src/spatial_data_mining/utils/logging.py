import logging
from typing import Dict, Any


def setup_logging(logging_cfg: Dict[str, Any]) -> None:
    level = logging_cfg.get("level", "INFO")
    fmt = logging_cfg.get("format", "[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
    logging.basicConfig(level=level, format=fmt)
