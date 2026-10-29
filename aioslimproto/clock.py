"""Track a player's clock (jiffies) against the server clock, as Lyrion does."""

from __future__ import annotations

from collections import deque

from .errors import PlayerClockUnknown

JIFFIES_OFFSET_TRACKING_LIST_SIZE = 50  # must be this big for large forward jumps
JIFFIES_OFFSET_TRACKING_LIST_MIN = 10  # must be this big to adjust at all
JIFFIES_EPOCH_MIN_ADJUST = 0.001  # seconds
JIFFIES_EPOCH_MAX_ADJUST = 0.005  # seconds
JIFFIES_EPOCH_RESET_JUMP = 50  # seconds; a larger jump means first sample or clock wrap
TICS_PER_SEC = 1000
JIFFIES_MASK = 0xFFFFFFFF


class JiffiesEpoch:
    """Server time (time.time()) at which the player's jiffies counter read zero.

    Port of Slim::Player::Player::trackJiffiesEpoch (Lyrion Music Server 9.2): the
    epoch is the lowest offset seen (a report that arrived with the least delay),
    adopted at once when lower, raised slowly when every recent sample sits above it.
    The tracker never reads a clock itself; the receive time is passed in.
    """

    def __init__(self) -> None:
        """Initialize the tracker without any sample."""
        self._epoch: float | None = None
        self._offsets: deque[float] = deque(maxlen=JIFFIES_OFFSET_TRACKING_LIST_SIZE)

    @property
    def epoch(self) -> float | None:
        """Return the epoch, None until the first sample."""
        return self._epoch

    def reset(self) -> None:
        """Forget the epoch and the samples (player reconnected)."""
        self._epoch = None
        self._offsets.clear()

    def track(self, jiffies: int, received_at: float) -> float:
        """Feed one status report; return how far its offset sits above the epoch."""
        offset = received_at - jiffies / TICS_PER_SEC
        if (
            self._epoch is None
            or offset < self._epoch  # simply a better estimate, or
            or offset - self._epoch
            > JIFFIES_EPOCH_RESET_JUMP  # wrap-around (or first time)
        ):
            self._epoch = offset
        diff = offset - self._epoch
        self._offsets.append(diff)
        if (
            diff > JIFFIES_EPOCH_MIN_ADJUST
            and len(self._offsets) >= JIFFIES_OFFSET_TRACKING_LIST_MIN
        ):
            min_diff = min(self._offsets)
            if min_diff > JIFFIES_EPOCH_MIN_ADJUST:
                # jumps larger than the max adjust need a full window of samples
                if (
                    min_diff > JIFFIES_EPOCH_MAX_ADJUST
                    and len(self._offsets) < JIFFIES_OFFSET_TRACKING_LIST_SIZE
                ):
                    return diff
                self._epoch += min_diff
                diff -= min_diff
                self._offsets.clear()  # start tracking again
        return diff

    def server_time(self, jiffies: int, packet_latency: float = 0.0) -> float:
        """Return the server time at which the player's clock read jiffies."""
        if self._epoch is None:
            raise PlayerClockUnknown
        return self._epoch + jiffies / TICS_PER_SEC - packet_latency

    def jiffies_at(self, server_time: float) -> int:
        """Return the player's clock at server_time as a u32 (for strm commands)."""
        if self._epoch is None:
            raise PlayerClockUnknown
        return int((server_time - self._epoch) * TICS_PER_SEC) & JIFFIES_MASK
