"""Ray actor pool with optional client-to-actor affinity."""

import ray


class ActorPool(ray.util.ActorPool):
    def get_actor_by_id(self, actor_id):
        if actor_id is None:
            return None
        actors = list(self._idle_actors) + [
            actor for _, actor in self._future_to_actor.values()
        ]
        for actor in actors:
            if actor_id in (actor._ray_actor_id, actor._ray_actor_id.hex()):
                return actor
        return None

    def submit(self, fn, value, affinity_actors=None):
        # Reserve the result index when submitted, even if affinity delays its
        # execution. Ray's ordered get_next() relies on these indices.
        task_index = self._next_task_index
        self._next_task_index += 1
        task = (fn, value, affinity_actors, task_index)
        if not self._start_task(*task):
            self._pending_submits.append(task)

    def _start_task(self, fn, value, affinity_actors, task_index):
        actors = self._idle_actors[-1:] if affinity_actors is None else affinity_actors
        for actor in actors:
            if actor in self._idle_actors:
                self._idle_actors.remove(actor)
                future = fn(actor, value)
                future_key = tuple(future) if isinstance(future, list) else future
                self._future_to_actor[future_key] = (task_index, actor)
                self._index_to_future[task_index] = future
                return True
        return False

    def _return_actor(self, actor):
        self._idle_actors.append(actor)
        # Ray's default FIFO retry can leave a compatible task stranded behind
        # a task bound to another busy actor. Start the first runnable task.
        for index, pending in enumerate(self._pending_submits):
            if self._start_task(*pending):
                self._pending_submits.pop(index)
                break
