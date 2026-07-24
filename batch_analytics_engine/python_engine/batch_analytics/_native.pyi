from typing import List

class Record:
    """A single numeric record in the stream."""

    id: str
    value: float
    timestamp: float

    def __init__(self, id: str, value: float, timestamp: float) -> None: ...

class AggregateResult:
    """Aggregation statistics for a fixed-size window of records."""

    window_start: int
    mean: float
    std: float
    outliers: List[int]

    def __init__(
        self, window_start: int, mean: float, std: float, outliers: List[int]
    ) -> None: ...

def aggregate_batch(records: List[Record], window: int) -> List[AggregateResult]:
    """Compute per-window mean and population standard deviation."""
    ...

def detect_outliers(results: List[AggregateResult], threshold: float) -> List[int]:
    """Return indices of windows whose mean is more than `threshold` std away."""
    ...

def validate_window(window: int) -> None:
    """Raise ValueError if window is not positive."""
    ...
