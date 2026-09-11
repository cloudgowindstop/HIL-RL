"""Rate controller."""

import time


class Rate:
    """The rate controller."""

    def __init__(self, rate: float) -> None:
        """Initialize the rate controller.

        Args:
            rate: The rate.
        """
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        """Sleep to maintain the rate."""
        target_time = self.last + 1.0 / self.rate
        remaining = target_time - time.time()
        if remaining > 0:
            # time.sleep(remaining)
            time.sleep(max(0, remaining - 0.001))  # Wake up 1 ms early
            while time.time() < target_time:       # Brief spin-wait
                pass
        self.last = time.time()
