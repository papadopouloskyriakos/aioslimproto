"""Tests for the player clock tracker."""

import pytest

from aioslimproto.clock import (
    JIFFIES_EPOCH_MAX_ADJUST,
    JIFFIES_OFFSET_TRACKING_LIST_MIN,
    JIFFIES_OFFSET_TRACKING_LIST_SIZE,
    JiffiesEpoch,
)
from aioslimproto.errors import PlayerClockUnknown

_U32 = 2**32


def _feed(
    tracker: JiffiesEpoch, count: int, *, offset: float, start: int = 1000
) -> None:
    """Feed count samples one second apart whose offset sits at the given value."""
    for index in range(count):
        jiffies = start + index * 1000
        tracker.track(jiffies, offset + jiffies / 1000)


def test_first_sample_sets_the_epoch() -> None:
    """The first report defines the epoch and sits exactly on it."""
    tracker = JiffiesEpoch()

    assert tracker.epoch is None
    assert tracker.track(1000, 100.0) == 0.0
    assert tracker.epoch == 99.0


def test_a_lower_offset_is_adopted_at_once() -> None:
    """A report that arrived faster is a better estimate and replaces the epoch."""
    tracker = JiffiesEpoch()
    tracker.track(1000, 100.0)

    tracker.track(2000, 100.9)

    assert tracker.epoch == pytest.approx(98.9)


def test_a_higher_offset_waits_while_the_adopting_sample_is_in_the_window() -> None:
    """The sample that set the epoch pins the window minimum at zero."""
    tracker = JiffiesEpoch()
    tracker.track(1000, 100.0)

    _feed(tracker, JIFFIES_OFFSET_TRACKING_LIST_SIZE - 1, offset=99.003, start=2000)

    assert tracker.epoch == 99.0


def test_small_drift_raises_the_epoch_once_the_window_is_clean() -> None:
    """Once every sample in the window sits above the epoch, it follows the minimum."""
    tracker = JiffiesEpoch()
    tracker.track(1000, 100.0)
    _feed(tracker, JIFFIES_OFFSET_TRACKING_LIST_SIZE, offset=99.003, start=2000)
    assert tracker.epoch == pytest.approx(99.003)

    # the window was cleared: a further raise needs the minimum sample count again
    _feed(tracker, JIFFIES_OFFSET_TRACKING_LIST_MIN - 1, offset=99.005, start=60_000)
    assert tracker.epoch == pytest.approx(99.003)
    _feed(tracker, 1, offset=99.005, start=70_000)
    assert tracker.epoch == pytest.approx(99.005)


def test_a_raise_above_the_max_adjust_waits_for_a_full_window() -> None:
    """A large raise is only made on a full window of samples."""
    tracker = JiffiesEpoch()
    tracker.track(1000, 100.0)
    _feed(tracker, JIFFIES_OFFSET_TRACKING_LIST_SIZE, offset=99.003, start=2000)
    offset = 99.003 + JIFFIES_EPOCH_MAX_ADJUST + 0.003

    _feed(tracker, JIFFIES_OFFSET_TRACKING_LIST_SIZE - 1, offset=offset, start=60_000)
    assert tracker.epoch == pytest.approx(99.003)
    _feed(tracker, 1, offset=offset, start=200_000)
    assert tracker.epoch == pytest.approx(offset)


def test_a_jump_over_the_reset_threshold_readopts() -> None:
    """A jiffies wrap-around makes the offset jump; the new value is adopted at once."""
    tracker = JiffiesEpoch()
    tracker.track(_U32 - 1000, 100.0)

    tracker.track(500, 101.5)

    assert tracker.epoch == pytest.approx(101.0)


def test_reset_forgets_epoch_and_samples() -> None:
    """After a reset the tracker behaves as if it never saw a report."""
    tracker = JiffiesEpoch()
    tracker.track(1000, 100.0)

    tracker.reset()

    assert tracker.epoch is None
    with pytest.raises(PlayerClockUnknown):
        tracker.server_time(1000)
    with pytest.raises(PlayerClockUnknown):
        tracker.jiffies_at(100.0)
    tracker.track(5000, 120.0)
    assert tracker.epoch == 115.0


def test_server_time_subtracts_packet_latency() -> None:
    """The report was read packet_latency after the player's clock read jiffies."""
    tracker = JiffiesEpoch()
    tracker.track(1000, 100.0)

    assert tracker.server_time(3000) == pytest.approx(102.0)
    assert tracker.server_time(3000, 0.002) == pytest.approx(101.998)


def test_jiffies_at_ignores_latency_and_masks_to_u32() -> None:
    """Targets encode on the player's clock as an unsigned 32-bit value."""
    tracker = JiffiesEpoch()
    tracker.track(1000, 100.0)

    assert tracker.jiffies_at(101.5) == 2500
    assert tracker.jiffies_at(99.0 + _U32 / 1000 + 0.204) == 204
    assert tracker.jiffies_at(98.0) == _U32 - 1000


def test_jittered_samples_settle_near_the_minimum_delay() -> None:
    """Reports with varying delivery delay settle the epoch on the fastest ones."""
    tracker = JiffiesEpoch()
    true_offset = 500.0
    for index in range(200):
        jiffies = 1000 + index * 1000
        jitter = (index * 7 % 5) / 1000  # 0 to 4 ms, deterministic
        tracker.track(jiffies, true_offset + jiffies / 1000 + jitter)

    assert tracker.epoch is not None
    assert abs(tracker.epoch - true_offset) < 0.001
