"""Small desktop multiprocessing helper."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def parallel_map(
    function: Callable[[InputT], OutputT],
    jobs: Iterable[InputT],
    *,
    workers: int | None = None,
) -> list[OutputT]:
    """Run independent jobs on local CPU cores and return ordered results."""
    job_list = list(jobs)
    if not job_list:
        return []

    worker_count = workers or min(len(job_list), os.cpu_count() or 1)
    if worker_count <= 1:
        return [function(job) for job in job_list]

    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        return list(pool.map(function, job_list))


__all__ = ["parallel_map"]
