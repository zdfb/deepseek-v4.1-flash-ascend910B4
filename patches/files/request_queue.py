# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
import os
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class TieredRequestQueue(RequestQueue):
    """把 WAITING 请求按 prompt 长度分成若干档（默认 4 档），档内 FCFS，
    档间按 V41_TIERED_ORDER 的顺序取（short_first 默认 / long_first）。

    - 档边界 V41_TIER_BOUNDS，逗号分隔，默认 "16384,65536,262144"
      ⇒ 档0 <16K、档1 16–64K、档2 64–256K、档3 ≥256K
    - 与 FCFS 的唯一差别是"先取哪一档"，不改变单请求语义
    """

    _bounds = None

    @staticmethod
    def _prompt_len(request):
        n = getattr(request, "num_prompt_tokens", None)
        if n is None:
            ids = getattr(request, "prompt_token_ids", None)
            n = len(ids) if ids is not None else 0
        return int(n)

    @classmethod
    def _get_bounds(cls):
        if cls._bounds is None:
            raw = os.environ.get("V41_TIER_BOUNDS", "16384,65536,262144")
            cls._bounds = [int(x) for x in raw.split(",") if x.strip()]
        return cls._bounds

    def __init__(self) -> None:
        self._tiers: list[deque] = [deque() for _ in range(len(self._get_bounds()) + 1)]
        if os.environ.get("V41_TIERED_ORDER", "short_first") == "long_first":
            self._order = list(range(len(self._tiers)))[::-1]
        else:
            self._order = list(range(len(self._tiers)))

    def _tier_of(self, request) -> int:
        n = self._prompt_len(request)
        bounds = self._get_bounds()
        for idx, bound in enumerate(bounds):
            if n < bound:
                return idx
        return len(bounds)

    def add_request(self, request: Request) -> None:
        self._tiers[self._tier_of(request)].append(request)

    def pop_request(self) -> Request:
        for idx in self._order:
            if self._tiers[idx]:
                return self._tiers[idx].popleft()
        raise IndexError("pop from an empty queue")

    def peek_request(self) -> Request:
        for idx in self._order:
            if self._tiers[idx]:
                return self._tiers[idx][0]
        raise IndexError("peek from an empty queue")

    def prepend_request(self, request: Request) -> None:
        self._tiers[self._tier_of(request)].appendleft(request)

    def prepend_requests(self, requests) -> None:
        for request in reversed(list(requests)):
            self._tiers[self._tier_of(request)].appendleft(request)

    def remove_request(self, request: Request) -> None:
        tier = self._tiers[self._tier_of(request)]
        try:
            tier.remove(request)
        except ValueError:
            pass

    def remove_requests(self, requests) -> None:
        target = requests if isinstance(requests, set) else set(requests)
        for tier in self._tiers:
            for request in list(tier):
                if request in target:
                    try:
                        tier.remove(request)
                    except ValueError:
                        pass

    def __bool__(self) -> bool:
        return any(self._tiers)

    def __len__(self) -> int:
        return sum(len(t) for t in self._tiers)

    def __iter__(self):
        for idx in self._order:
            yield from list(self._tiers[idx])


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if os.environ.get("V41_TIERED_SCHED", "0") not in ("", "0"):
        return TieredRequestQueue()
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
