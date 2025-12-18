from typing import Callable, Optional


class PipelineCancelled(Exception):
    """Raised when a cooperative stop is requested."""


def check_cancelled(should_stop: Optional[Callable[[], bool]]) -> None:
    """
    Raise PipelineCancelled if the provided callback returns True.
    """
    if should_stop and should_stop():
        raise PipelineCancelled("Pipeline stop requested by user.")
