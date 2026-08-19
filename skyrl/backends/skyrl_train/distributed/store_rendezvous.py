"""Bounded CPU-Store rendezvous for configuration decisions made before NCCL PG creation."""

from __future__ import annotations

import pickle
from datetime import timedelta


class StoreRendezvous:
    """Generate fresh Store namespaces across repeated setup calls in one process group."""

    def __init__(self) -> None:
        self._local_generations: dict[str, int] = {}

    def all_gather(self, store, rank: int, world: int, tag: str, value, timeout_seconds: int):
        local_generation = self._local_generations.get(tag, 0) + 1
        self._local_generations[tag] = local_generation
        root = f"skyrl/{tag}"
        announce_key = f"{root}/announce/{local_generation}"
        timeout = timedelta(seconds=timeout_seconds)
        if rank == 0:
            store_epoch = store.add(f"{root}/epoch", 1)
            store.set(announce_key, str(store_epoch).encode())
        store.wait([announce_key], timeout)
        store_epoch = int(store.get(announce_key).decode())

        prefix = f"{root}/epoch-{store_epoch}"
        store.set(f"{prefix}/{rank}", pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        keys = [f"{prefix}/{peer}" for peer in range(world)]
        store.wait(keys, timeout)
        return [pickle.loads(store.get(key)) for key in keys]


_RENDEZVOUS = StoreRendezvous()


def store_all_gather(store, rank: int, world: int, tag: str, value, timeout_seconds: int):
    return _RENDEZVOUS.all_gather(store, rank, world, tag, value, timeout_seconds)
