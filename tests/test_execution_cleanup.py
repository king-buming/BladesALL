"""Compatibility and scheduling regressions for the execution cleanup.

These checks use CPU tensors and fake task results; they do not start Ray.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import ray

from blades.core.execution.actor_manager import ActorManager, _ActorPool
from blades.utils import ActorPool
from blades.utils import torch_utils


class FakeActor:
    def __init__(self):
        self._ray_actor_id = ray.ActorID.from_random()


class ActorPoolCleanupTests(unittest.TestCase):
    def test_public_pool_and_manager_share_implementation(self):
        actor = FakeActor()
        manager = ActorManager(actors=[actor])
        self.assertIs(_ActorPool, ActorPool)
        self.assertIs(type(manager._actor_pool), ActorPool)
        self.assertIs(manager._get_actor_by_id(actor._ray_actor_id.hex()), actor)

    def test_lookup_finds_busy_actors_and_accepts_both_id_forms(self):
        actor = FakeActor()
        pool = ActorPool([actor])
        pool.submit(lambda actor, value: value, "pending", [actor])
        self.assertIs(pool.get_actor_by_id(actor._ray_actor_id), actor)
        self.assertIs(pool.get_actor_by_id(actor._ray_actor_id.hex()), actor)
        self.assertIsNone(pool.get_actor_by_id(None))
        self.assertIsNone(pool.get_actor_by_id("unknown"))

    def test_affinity_actor_is_busy_until_its_result_is_consumed(self):
        actor = FakeActor()
        pool = ActorPool([actor])
        started = []

        def task(target, value):
            started.append(value)
            return value

        pool.submit(task, "first", [actor])
        pool.submit(task, "second", [actor])
        self.assertEqual(started, ["first"])
        self.assertFalse(pool.has_free())
        with patch("ray.get", side_effect=lambda result: result):
            self.assertEqual(pool.get_next(), "first")
            self.assertEqual(started, ["first", "second"])
            self.assertEqual(pool.get_next(), "second")
        self.assertFalse(pool.has_next())
        self.assertTrue(pool.has_free())

    def test_pending_affinity_tasks_are_not_stranded_behind_another_actor(self):
        first, second = FakeActor(), FakeActor()
        pool = ActorPool([first, second])
        started = []

        def task(actor, value):
            started.append((actor, value))
            return value

        pool.submit(task, "first-0", [first])
        pool.submit(task, "second-0", [second])
        pool.submit(task, "second-1", [second])
        pool.submit(task, "first-1", [first])
        with patch("ray.get", side_effect=lambda result: result):
            results = []
            while pool.has_next():
                results.append(pool.get_next())
        self.assertEqual(results, ["first-0", "second-0", "second-1", "first-1"])
        self.assertEqual(len(started), 4)
        self.assertEqual(pool._pending_submits, [])
        self.assertEqual(pool._idle_actors.count(first), 1)
        self.assertEqual(pool._idle_actors.count(second), 1)

    def test_later_task_on_free_actor_does_not_overtake_earlier_queued_result(self):
        first, second = FakeActor(), FakeActor()
        pool = ActorPool([first, second])
        started = []

        def task(actor, value):
            started.append(value)
            return value

        pool.submit(task, "first-0", [first])
        pool.submit(task, "first-1", [first])
        pool.submit(task, "other-free")
        self.assertEqual(started, ["first-0", "other-free"])
        with patch("ray.get", side_effect=lambda result: result):
            self.assertEqual(
                [pool.get_next(), pool.get_next(), pool.get_next()],
                ["first-0", "first-1", "other-free"],
            )

    def test_unordered_results_keep_queued_indices_and_reject_switching_to_ordered(self):
        first, second = FakeActor(), FakeActor()
        pool = ActorPool([first, second])
        task = lambda actor, value: value
        pool.submit(task, "first-0", [first])
        pool.submit(task, "first-1", [first])
        pool.submit(task, "second-0", [second])
        with (
            patch("ray.get", side_effect=lambda result: result),
            patch("ray.wait", side_effect=lambda refs, **kwargs: ([refs[-1]], refs[:-1])),
        ):
            self.assertEqual(pool.get_next_unordered(), "second-0")
            with self.assertRaisesRegex(ValueError, "get_next_unordered"):
                pool.get_next()
            self.assertEqual(pool.get_next_unordered(), "first-0")
            self.assertEqual(pool.get_next_unordered(), "first-1")
        self.assertFalse(pool.has_next())
        self.assertEqual(pool._pending_submits, [])
        self.assertEqual(pool._index_to_future, {})

    def test_default_and_affinity_submissions_can_share_the_pending_queue(self):
        actor = FakeActor()
        pool = ActorPool([actor])
        pool.submit(lambda actor, value: value, "default")
        pool.submit(lambda actor, value: value, "affinity", [actor])
        pool.submit(lambda actor, value: value, "queued-default")
        with patch("ray.get", side_effect=lambda result: result):
            self.assertEqual(
                [pool.get_next(), pool.get_next(), pool.get_next()],
                ["default", "affinity", "queued-default"],
            )


class TensorCompatibilityTests(unittest.TestCase):
    def test_vector_round_trip_preserves_frozen_parameters_and_storage_semantics(self):
        first = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        frozen = torch.nn.Parameter(torch.tensor([99.0]), requires_grad=False)
        last = torch.nn.Parameter(torch.tensor([[3.0, 4.0]]))
        parameters = [first, frozen, last]
        torch.testing.assert_close(
            torch_utils.parameters_to_vector(iter(parameters)),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        replacement = torch.tensor([5.0, 6.0, 7.0, 8.0])
        torch_utils.vector_to_parameters(replacement, iter(parameters))
        torch.testing.assert_close(frozen, torch.tensor([99.0]))
        torch.testing.assert_close(first, torch.tensor([5.0, 6.0]))
        torch.testing.assert_close(last, torch.tensor([[7.0, 8.0]]))
        self.assertEqual(first.data_ptr(), replacement.data_ptr())

    def test_update_clipping_returns_tensor_and_keeps_input_storage(self):
        update = torch.tensor([3.0, 4.0], requires_grad=True)
        clipped = torch_utils.clip_tensor_norm_(update, 1.0)
        torch.testing.assert_close(clipped, torch.tensor([0.6, 0.8]))
        self.assertEqual(clipped.data_ptr(), update.data_ptr())
        self.assertFalse(clipped.requires_grad)
        self.assertTrue(update.requires_grad)

    def test_legacy_iterable_clipping_only_changes_and_returns_first_float_tensor(self):
        first, second = torch.tensor([3.0]), torch.tensor([4.0])
        result = torch_utils.clip_tensor_norm_([first, second], 1.0)
        torch.testing.assert_close(result, torch.tensor([0.6]))
        torch.testing.assert_close(second, torch.tensor([4.0]))
        self.assertEqual(result.data_ptr(), first.data_ptr())

    def test_parameter_mapping_clips_all_float_entries_and_returns_original_norm(self):
        parameters = {
            "first": torch.tensor([3.0]),
            "counter": torch.tensor(7, dtype=torch.int64),
            "second": torch.tensor([4.0]),
        }
        norm = torch_utils.clip_para_norm_(parameters, 1.0)
        torch.testing.assert_close(norm, torch.tensor(5.0))
        torch.testing.assert_close(parameters["first"], torch.tensor([0.6]))
        torch.testing.assert_close(parameters["second"], torch.tensor([0.8]))
        self.assertEqual(parameters["counter"].item(), 7)

    def test_clipping_handles_empty_zero_infinite_norm_and_nonfinite_rejection(self):
        torch.testing.assert_close(torch_utils.clip_tensor_norm_([], 1), torch.tensor(0.0))
        zero = torch.zeros(2)
        torch.testing.assert_close(torch_utils.clip_tensor_norm_(zero, 1), zero)
        update = torch.tensor([-3.0, 4.0])
        torch.testing.assert_close(
            torch_utils.clip_tensor_norm_(update, 2.0, norm_type=torch.inf),
            torch.tensor([-1.5, 2.0]),
        )
        update = torch.tensor([float("nan"), 1.0])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            torch_utils.clip_tensor_norm_(update, 1, error_if_nonfinite=True)
        self.assertEqual(update[1].item(), 1.0)

    def test_gradient_clipping_retains_zero_norm_empty_result(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        policy = SimpleNamespace(config={"grad_clip": 1.0})
        self.assertEqual(torch_utils.apply_grad_clipping(policy, optimizer, None), {})
        parameter.grad = torch.zeros_like(parameter)
        self.assertEqual(torch_utils.apply_grad_clipping(policy, optimizer, None), {})
        parameter.grad = torch.tensor([3.0, 4.0])
        result = torch_utils.apply_grad_clipping(policy, optimizer, None)
        self.assertAlmostEqual(float(result["grad_gnorm"]), 1.0)
        torch.testing.assert_close(parameter.grad, torch.tensor([0.6, 0.8]))

    def test_legacy_device_helper_keeps_cpu_and_insufficient_gpu_behaviour(self):
        config = {"_fake_gpus": False, "num_gpus": 0, "num_gpus_per_worker": 0}
        with patch("torch.cuda.device_count", return_value=0):
            self.assertEqual(torch_utils.get_device(config), torch.device("cpu"))
        config["num_gpus"] = 2
        with (
            patch("torch.cuda.device_count", return_value=1),
            patch("ray._private.worker._mode", return_value=0),
        ):
            with self.assertRaisesRegex(ValueError, "enough GPU IDs"):
                torch_utils.get_device(config)


class LegacyModuleCompatibilityTests(unittest.TestCase):
    def test_registry_aliases_keep_one_table_and_distinct_fedavg_names(self):
        from blades.algorithms import registry
        from blades.algorithms.fedavg import Fedavg
        from blades.trainers import registry as legacy
        from blades.trainers.fedavg import FedavgTrainer

        self.assertIs(legacy.TRAINERS, registry.ALGORITHMS)
        self.assertIs(legacy._get_algorithm_class, registry._get_algorithm_class)
        self.assertIs(legacy._import_fedavg()[0], Fedavg)
        self.assertIs(legacy.TRAINERS["FEDAVG_BASE"]()[0], FedavgTrainer)
        self.assertIs(legacy._import_fedprox()[0], registry._import_fedprox()[0])

    def test_legacy_session_initialization_and_worker_delegation_remain_available(self):
        from blades import session

        worker = SimpleNamespace(
            get_global_model=lambda from_state: ("model", from_state),
            get_global_optimizer=lambda: "optimizer",
        )
        with patch.object(session, "_fl_session", None):
            with self.assertRaisesRegex(Exception, "not initialized"):
                session.get_session()
            with self.assertWarns(DeprecationWarning):
                session.init_session()
            self.assertIsInstance(session.get_session(), session.FLSession)
            with patch.object(session, "get_current_worker", return_value=worker):
                self.assertEqual(session.get_global_model(True), ("model", True))
                self.assertEqual(session.get_global_optimizer(), "optimizer")

    def test_legacy_queue_warns_and_keeps_sync_protocol_without_starting_ray(self):
        from blades.utils.queue import Queue, RayQueue

        with patch.object(RayQueue, "__init__", return_value=None) as initialize:
            with self.assertWarns(DeprecationWarning):
                queue = Queue(3, {"num_cpus": 0})
            initialize.assert_called_once_with(3, {"num_cpus": 0})
        value = torch.tensor([1.0])
        with patch.object(RayQueue, "put") as put:
            queue.put(value, block=False)
            put.assert_called_once_with(value, False, None)
        with patch.object(RayQueue, "get", return_value=value):
            self.assertIs(queue.get(), value)
        with patch.object(RayQueue, "get", return_value=(sum, ([1, 2],))):
            self.assertEqual(queue.get(), 3)


if __name__ == "__main__":
    unittest.main()
