"""Real-core contracts for the spectral attack's trainer callback."""

import importlib
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from blades.constants import CLIENT_ID, CLIENT_UPDATE


MODULE = "blades.adversaries.spear_spectral_adversary"
SMALL_SEARCH = dict(
    z_values=(0.0, 0.2, 0.5),
    spectral_penalties=(0.0, 3.0),
    spectral_rank=1,
    fit_subspaces=2,
    dnc_sub_dim=3,
    dnc_num_iters=2,
    search_banks=2,
    audit_banks=2,
    seed=43,
)


def attack_class():
    assert importlib.util.find_spec(MODULE) is not None, "spectral adversary is missing"
    return importlib.import_module(MODULE).SPEARSpectralAdversary


def toy_benign():
    generator = torch.Generator().manual_seed(203)
    return torch.randn(4, 6, generator=generator, dtype=torch.float64) + 0.3


def test_module_path_config_constructs_an_attack_that_generates_finite_updates():
    from ray.rllib.utils.from_config import from_config

    cls = attack_class()
    attack = from_config({"type": MODULE + ".SPEARSpectralAdversary", **SMALL_SEARCH})
    assert isinstance(attack, cls)
    benign = toy_benign()
    update = attack.generate_update(benign, 1)
    assert update.shape == (6,)
    assert update.dtype == benign.dtype
    assert update.device == benign.device
    assert torch.isfinite(update).all()
    assert attack.last_diagnostics
    assert attack.round_index == 1


def test_round_uses_participating_count_and_real_submission_order():
    from blades.adversaries.spectral_spear import search_spectral_update

    attack = attack_class()(**SMALL_SEARCH)
    benign = toy_benign()
    # Configured population includes an absent malicious client (ID 2).
    malicious_ids = {2, 8, 9}
    order = [4, 9, 1, 6, 7, 8, 5]
    benign_ids = [4, 1, 7, 5]
    originals = {uid: benign[i].clone() for i, uid in enumerate(benign_ids)}
    clients = {uid: SimpleNamespace(is_malicious=uid in malicious_ids)
               for uid in order + [2]}
    results = [{CLIENT_ID: uid, CLIENT_UPDATE: originals.get(uid)} for uid in order]
    # The base adversary disables malicious local training, returning ID only.
    del results[1][CLIENT_UPDATE]
    trainer = SimpleNamespace(
        local_results=results,
        config=SimpleNamespace(num_clients=20, num_malicious_clients=3),
        client_manager=SimpleNamespace(get_client_by_id=clients.__getitem__),
    )
    expected, report = search_spectral_update(
        benign, 2, config=attack.spectral_config, round_index=0,
        malicious_positions=[1, 4],
    )

    attack.on_local_round_end(trainer)

    assert attack.last_diagnostics == report
    assert attack.round_index == 1
    assert [r[CLIENT_ID] for r in results] == order
    assert results[3][CLIENT_UPDATE] is None
    for i, uid in enumerate(benign_ids):
        result = next(r for r in results if r[CLIENT_ID] == uid)
        assert result[CLIENT_UPDATE] is originals[uid]
        torch.testing.assert_close(result[CLIENT_UPDATE], benign[i], rtol=0, atol=0)
    submitted = [r[CLIENT_UPDATE] for r in results if r[CLIENT_ID] in malicious_ids]
    for update in submitted:
        torch.testing.assert_close(update, expected, rtol=0, atol=0)
    assert submitted[0].data_ptr() != submitted[1].data_ptr()
    submitted[0].add_(10)
    torch.testing.assert_close(submitted[1], expected, rtol=0, atol=0)


def test_no_active_malicious_clients_is_a_noop_without_benign_data():
    attack = attack_class()(**SMALL_SEARCH)
    trainer = SimpleNamespace(local_results=[], client_manager=SimpleNamespace())
    attack.on_local_round_end(trainer)
    assert attack.round_index == 0
    assert attack.last_diagnostics == {}


def test_no_active_malicious_clients_preserves_benign_results():
    attack = attack_class()(**SMALL_SEARCH)
    original = torch.tensor([1.0, -2.0])
    results = [{CLIENT_ID: 4, CLIENT_UPDATE: original}]
    trainer = SimpleNamespace(
        local_results=results,
        client_manager=SimpleNamespace(
            get_client_by_id=lambda _: SimpleNamespace(is_malicious=False)),
    )
    attack.on_local_round_end(trainer)
    assert results[0][CLIENT_UPDATE] is original
    torch.testing.assert_close(original, torch.tensor([1.0, -2.0]))
    assert attack.round_index == 0


def test_invalid_input_keeps_round_state_and_last_successful_diagnostics():
    attack = attack_class()(**SMALL_SEARCH)
    attack.generate_update(toy_benign(), 1)
    previous = attack.last_diagnostics.copy()
    with pytest.raises((ValueError, TypeError)):
        attack.generate_update(torch.tensor([[float("nan"), 1.0]]), 1)
    assert attack.round_index == 1
    assert attack.last_diagnostics == previous


def test_invalid_configuration_is_rejected_by_the_wrapper():
    with pytest.raises(ValueError):
        attack_class()(min_survival=0.0)


def test_infeasible_proxy_returns_finite_mean_with_honest_status():
    attack = attack_class()(z_values=(0.0,), spectral_penalties=(1.0,),
                           dnc_num_iters=1, search_banks=1, audit_banks=1)
    benign = torch.zeros(2, 3)
    update = attack.generate_update(benign, 2, malicious_positions=[2, 3])
    torch.testing.assert_close(update, torch.zeros(3))
    assert attack.last_diagnostics["status"] == "no_feasible_candidate"
    assert attack.last_diagnostics["selected"] is None
    assert attack.round_index == 1


def test_generate_update_preserves_global_rng_streams_and_replays_same_round():
    cls = attack_class()
    first, replay = cls(**SMALL_SEARCH), cls(**SMALL_SEARCH)
    benign = toy_benign()
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    for round_index in range(2):
        update = first.generate_update(benign, 2, malicious_positions=[0, 3])
        other = replay.generate_update(benign, 2, malicious_positions=[0, 3])
        torch.testing.assert_close(update, other, rtol=0, atol=0)
        assert first.last_diagnostics == replay.last_diagnostics
        assert first.round_index == replay.round_index == round_index + 1

    assert torch.equal(torch_state, torch.random.get_rng_state())
    actual_numpy = np.random.get_state()
    assert numpy_state[0] == actual_numpy[0]
    assert np.array_equal(numpy_state[1], actual_numpy[1])
    assert numpy_state[2:] == actual_numpy[2:]
    assert random.getstate() == python_state


def test_smoke_trains_two_cpu_rounds_with_spectral_attack_and_real_dnc():
    module_name = "examples.smoke_spectral_spear"
    assert importlib.util.find_spec(module_name) is not None, "spectral smoke example is missing"
    results = importlib.import_module(module_name).run_smoke(rounds=2)
    assert [row["round"] for row in results] == [1, 2]
    for row in results:
        assert np.isfinite(row["loss"])
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["attack_status"] in ("proxy_feasible", "no_feasible_candidate")
