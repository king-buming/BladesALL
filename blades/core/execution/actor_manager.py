from typing import Any, Callable, Optional, List

from ray.actor import ActorHandle
from ray.rllib.utils.actor_manager import FaultTolerantActorManager

from blades.utils.actor_pool import ActorPool


# Historical private import path; both entry points share one implementation.
_ActorPool = ActorPool


class ActorManager(FaultTolerantActorManager):
    def __init__(
        self,
        actors: Optional[List[ActorHandle]] = None,
        max_remote_requests_in_flight_per_actor: Optional[int] = 2,
        init_id: Optional[int] = 0,
    ):
        super().__init__(
            actors,
            max_remote_requests_in_flight_per_actor,
            init_id,
        )

        self._actor_pool = ActorPool(self.actors().values())

    def _get_actor_by_id(self, actor_id: str) -> ActorHandle:
        return self._actor_pool.get_actor_by_id(actor_id)

    def execute_with_actor_pool(
        self,
        func: Callable,
        values: List[Any],
        affinity_actors: List[List[str]] = None,
    ):
        if not affinity_actors:
            affinity_actors = [None for _ in range(len(values))]

        elif len(values) != len(affinity_actors):
            raise ValueError(
                f"Number of values ({len(values)}) does not match number of"
                f"affinity_actors ({len(affinity_actors)})"
            )
        for value, actor_ids in zip(values, affinity_actors):
            if actor_ids is None:
                self._actor_pool.submit(
                    lambda a, value: a.apply.remote(func, value), value
                )
            else:
                target_actors = [
                    self._get_actor_by_id(actor_id) for actor_id in actor_ids
                ]
                self._actor_pool.submit(
                    lambda a, value: a.apply.remote(func, value),
                    value,
                    affinity_actors=target_actors,
                )

        results = []
        while self._actor_pool.has_next():
            client_result = self._actor_pool.get_next()
            results.append(client_result)
        return results
