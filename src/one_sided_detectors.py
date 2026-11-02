"""Python ports of the SEED and STEPD drift detectors, with an optional one-sided test.

MOA's SEED and STEPD flag a change when the error goes up or down. A fall in the error
is not a drift in our streams (a drift is a new task, which raises the error), so with
`one_sided=True` they fire only when the recent error is higher than the older error.

With `one_sided=False` the ports follow MOA step by step (see `tests/test_one_sided_
detectors.py`, which checks the alarms against the Java detectors). The input is an
error bit: 1 for a wrong prediction.
"""

from __future__ import annotations

from math import erfc, log, sqrt

from capymoa.drift.base_detector import BaseDriftDetector


def _normal_cdf(z: float) -> float:
    return 0.5 * erfc(-z / sqrt(2.0))


class _Detector(BaseDriftDetector):
    """The part of the CapyMOA detector interface that the experiments use."""

    def __init__(self) -> None:
        super().__init__()
        self.in_concept_change = False
        self.in_warning_zone = False

    def reset(self, clean_history: bool = False) -> None:
        super().reset(clean_history)
        self._reset_state()

    def _reset_state(self) -> None:
        raise NotImplementedError

    def _record(self, change: bool, warning: bool) -> None:
        self.idx += 1
        self.in_concept_change = change
        self.in_warning_zone = warning
        if warning:
            self.warning_index.append(self.idx)
        if change:
            self.detection_index.append(self.idx)


class STEPD(_Detector):
    """Statistical Test of Equal Proportions (Nishida and Yamauchi, 2007).

    Compares the error in the last `window_size` instances with the error in all older
    instances since the last change, with a z-test of two proportions.
    """

    def __init__(
        self,
        window_size: int = 30,
        alpha_drift: float = 0.003,
        alpha_warning: float = 0.05,
        one_sided: bool = True,
    ) -> None:
        self.window_size = window_size
        self.alpha_drift = alpha_drift
        self.alpha_warning = alpha_warning
        self.one_sided = one_sided
        super().__init__()
        self._reset_state()

    def _reset_state(self) -> None:
        self._stored = [0] * self.window_size  # ring buffer of the recent errors
        self._first = 0
        self._last = -1
        self._w_old = 0.0  # errors in the older part
        self._w_recent = 0.0  # errors in the recent part
        self._n_old = 0
        self._n_recent = 0
        self._change = False

    def add_element(self, element: float) -> None:
        # MOA clears its state on the input after a change
        if self._change:
            self._reset_state()
        if self._n_recent == self.window_size:
            # The oldest recent instance moves to the older part
            moved = self._stored[self._first]
            self._w_old += moved
            self._n_old += 1
            self._w_recent -= moved
            self._first += 1
            if self._first == self.window_size:
                self._first = 0
        else:
            self._n_recent += 1
        self._last += 1
        if self._last == self.window_size:
            self._last = 0
        self._stored[self._last] = int(element)
        self._w_recent += element

        change = warning = False
        if self._n_old >= self.window_size:
            n_o, n_r = self._n_old, self._n_recent
            # The error rates, and their difference (positive when the error rose)
            diff = self._w_recent / n_r - self._w_old / n_o
            p = (n_o - self._w_old + n_r - self._w_recent) / (n_o + n_r)
            inv_sum = 1.0 / n_o + 1.0 / n_r
            spread = sqrt(p * (1.0 - p) * inv_sum)
            if spread > 0.0:
                if self.one_sided:
                    z = (diff - inv_sum / 2.0) / spread
                    p_value = 1.0 - _normal_cdf(z)
                else:
                    z = (abs(diff) - inv_sum / 2.0) / spread
                    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
            elif self.one_sided:
                p_value = 1.0  # Every bit is the same, so the error did not rise
            else:
                # MOA divides by zero here, gets an infinite z and flags a change
                p_value = 0.0
            change = p_value < self.alpha_drift
            warning = not change and p_value < self.alpha_warning
        self._change = change
        self._record(change, warning)


class _Block:
    __slots__ = ("next", "previous", "block_size", "total", "variance", "item_count")

    def __init__(self, block_size: int) -> None:
        self.next: _Block | None = None
        self.previous: _Block | None = None
        self.block_size = block_size
        self.total = 0.0
        self.variance = 0.0
        self.item_count = 0

    def is_full(self) -> bool:
        return self.item_count == self.block_size


class _Window:
    """The SEED window: a list of blocks, the oldest first. Linear decay, fixed term."""

    def __init__(
        self, block_size: int, epsilon_prime: float, alpha: float, compress_term: int
    ) -> None:
        self.block_size = block_size
        self.epsilon_prime = epsilon_prime
        self.alpha = alpha
        self.compress_term = compress_term
        self.head: _Block = _Block(block_size)
        self.tail: _Block = self.head
        self.width = 0
        self.block_count = 1
        self.total = 0.0
        self.variance = 0.0
        self.compress_count = 0

    def add(self, value: float) -> None:
        if self.tail.is_full():
            if self.tail.previous is not None and self.compress_count > self.compress_term:
                self.compress_count = 0
                cursor: _Block | None = self.tail
                epsilon = 0.0
                while cursor is not None and cursor.previous is not None:
                    n0, n1 = cursor.item_count, cursor.previous.item_count
                    u0, u1 = cursor.total, cursor.previous.total
                    diff = abs(u1 / n1 - u0 / n0)
                    epsilon += self.epsilon_prime * self.alpha
                    if diff < epsilon:
                        self._compress(cursor)
                    cursor = cursor.previous
            block = _Block(self.block_size)
            block.previous = self.tail
            self.tail.next = block
            self.tail = block
            self.block_count += 1
            self.compress_count += 1
        self.tail.item_count += 1
        self.tail.total += value
        self.total += value
        self.width += 1
        if self.width >= 2:
            # As in MOA, the total already holds the new value here
            inc = (
                (self.width - 1)
                * (value - self.total / (self.width - 1)) ** 2
                / self.width
            )
            self.variance += inc
            self.tail.variance += inc

    def _compress(self, cursor: _Block) -> None:
        """Merge a block into the one before it."""
        previous = cursor.previous
        previous.total += cursor.total
        previous.item_count += cursor.item_count
        previous.variance += cursor.variance
        previous.block_size += cursor.block_size
        if cursor.next is not None:
            previous.next = cursor.next
            cursor.next.previous = previous
        else:
            previous.next = None
            self.tail = previous
        self.block_count -= 1


class SEED(_Detector):
    """SEED (Huang, Koh, Dobbie and Pears, 2014): an ADWIN-like test on blocks of data."""

    def __init__(
        self,
        delta: float = 0.05,
        block_size: int = 32,
        epsilon_prime: float = 0.01,
        alpha: float = 0.8,
        compress_term: int = 75,
        one_sided: bool = True,
    ) -> None:
        self.delta = delta
        self.block_size = block_size
        self.epsilon_prime = epsilon_prime
        self.alpha = alpha
        self.compress_term = compress_term
        self.one_sided = one_sided
        super().__init__()
        self._reset_state()

    def _reset_state(self) -> None:
        self._window = _Window(
            self.block_size, self.epsilon_prime, self.alpha, self.compress_term
        )
        self._count = 0

    def _bound(self, n0: float, n1: float) -> float:
        w = self._window
        n = n0 + n1
        dd = log(2 * log(n) / self.delta)
        v = w.variance / w.width
        m = 1 / n0 + 1 / n1
        return sqrt(2 * m * v * dd) + 2.0 / 3.0 * dd * m

    def add_element(self, element: float) -> None:
        self._window.add(element)
        self._count += 1
        self._record(self._test(), False)

    def _test(self) -> bool:
        w = self._window
        if self._count % self.block_size != 0 or w.block_count < 2:
            return False
        n1, n0 = 0, w.width
        u1, u0 = 0.0, w.total
        cursor = w.tail
        # Cut the window in two at each block from the newest back. The newer part
        # (n1, u1) is compared with the older part (n0, u0).
        while cursor.previous is not None:
            n0 -= cursor.item_count
            n1 += cursor.item_count
            u0 -= cursor.total
            u1 += cursor.total
            diff = u1 / n1 - u0 / n0  # positive when the error rose
            if not self.one_sided:
                diff = abs(diff)
            if diff > self._bound(n0, n1):
                # Drop everything before this block
                w.head = cursor
                while cursor.previous is not None:
                    cursor = cursor.previous
                    w.width -= cursor.item_count
                    w.total -= cursor.total
                    w.variance -= cursor.variance
                    w.block_count -= 1
                w.head.previous = None
                return True
            cursor = cursor.previous
        return False
