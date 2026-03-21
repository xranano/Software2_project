from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SPEED = 0.5
TURN = 0.3
MIN_SPEED = -1.0
MAX_SPEED = 1.0


def clamp(value: float, min_value: float = MIN_SPEED, max_value: float = MAX_SPEED) -> float:
    """Ensure motor speeds stay within safe limits."""
    return max(min(value, max_value), min_value)


def get_motor_speeds(keys_pressed: Dict[str, bool]) -> Tuple[float, float]:

    linear = 0.0   # forward/backward
    angular = 0.0  # turning

    # Linear movement
    if keys_pressed.get('up'):
        linear += SPEED
    if keys_pressed.get('down'):
        linear -= SPEED

    # Angular movement
    if keys_pressed.get('left'):
        angular += TURN
    if keys_pressed.get('right'):
        angular -= TURN

    # Convert to motor speeds
    left = linear - angular
    right = linear + angular

    # Clamp values
    left = clamp(left)
    right = clamp(right)

    logger.debug(f"Keys: {keys_pressed} -> Left: {left}, Right: {right}")

    return left, right

