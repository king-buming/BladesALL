"""Trainer integration for SPEAR's spectral candidate optimization."""

import torch

from blades.constants import CLIENT_ID, CLIENT_UPDATE

from .adversary import Adversary
from .spectral_spear import SpectralSpearConfig, search_spectral_update


class SPEARSpectralAdversary(Adversary):
    """Optimize colluding uploads with full knowledge of available benign updates.

    The configured DnC parameters describe an attacker-side proxy. Screening and
    held-out audit results are recorded in ``last_diagnostics``; they do not
    guarantee acceptance under the server's independently sampled coordinates.
    The original ``SPEARAdversary`` remains available as a separate baseline.
    """

    def __init__(
        self,
        z_values=(0.0, 0.5, 1.0, 2.0, 4.0),
        spectral_penalties=(0.0, 1.0, 10.0),
        spectral_rank=2,
        fit_subspaces=3,
        dnc_sub_dim=10000,
        dnc_num_iters=5,
        search_banks=2,
        audit_banks=4,
        dnc_filter_frac=1.0,
        seed=2026,
        min_survival=1.0,
    ):
        super().__init__()
        self.spectral_config = SpectralSpearConfig(
            z_values=z_values,
            spectral_penalties=spectral_penalties,
            spectral_rank=spectral_rank,
            fit_subspaces=fit_subspaces,
            dnc_sub_dim=dnc_sub_dim,
            dnc_num_iters=dnc_num_iters,
            search_banks=search_banks,
            audit_banks=audit_banks,
            dnc_filter_frac=dnc_filter_frac,
            seed=seed,
            min_survival=min_survival,
        )
        self.last_diagnostics = {}
        self.round_index = 0

    def generate_update(self, benign_updates, num_malicious, malicious_positions=None):
        """Generate one upload; advance the local seed round only after success."""
        update, diagnostics = search_spectral_update(
            benign_updates,
            num_malicious,
            config=self.spectral_config,
            round_index=self.round_index,
            malicious_positions=malicious_positions,
        )
        self.last_diagnostics = diagnostics
        self.round_index += 1
        return update

    def on_local_round_end(self, trainer):
        """Replace participating malicious uploads while preserving result order."""
        benign_updates = []
        malicious_results = []
        malicious_positions = []
        position = 0
        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                # Base Adversary disables their local training: an ID-only result
                # still represents a participating client requiring an upload.
                malicious_results.append(result)
                malicious_positions.append(position)
                position += 1
            elif result.get(CLIENT_UPDATE) is not None:
                benign_updates.append(result[CLIENT_UPDATE])
                position += 1

        if not malicious_results:
            return
        if not benign_updates:
            raise ValueError("No benign updates found; spectral SPEAR needs benign visibility.")

        update = self.generate_update(
            torch.vstack(benign_updates),
            len(malicious_results),
            malicious_positions=malicious_positions,
        )
        for result in malicious_results:
            result[CLIENT_UPDATE] = update.clone()
