"""Module for reconstructing neutrino direction from primary particles."""

import numpy as np

from spine.post.base import PostBase

__all__ = ["NeutrinoDirectionProcessor"]


class NeutrinoDirectionProcessor(PostBase):
    """Reconstructs the neutrino momentum and direction for each interaction.

    This module computes the sum of the momenta of all primary particles
    associated with an interaction to estimate the neutrino momentum and direction.
    """

    # Name of the post-processor
    name = "nu_direction"

    # Alternative allowed names
    aliases = ("neutrino_direction",)

    def __init__(self, run_mode="reco"):
        """Initialize the neutrino direction processor.

        Parameters
        ----------
        run_mode : str, default 'reco'
            Determines whether to run on reconstructed, truth, or both objects.
        """
        # Initialize the parent class for interactions
        super().__init__("interaction", run_mode)

    def process(self, data):
        """Update each interaction with reconstructed neutrino momentum.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        for key in self.interaction_keys:
            for interaction in data[key]:
                nu_momentum = np.zeros(3, dtype=np.float32)
                valid_mom_count = 0
                for part in interaction.primary_particles:
                    mom = part.momentum
                    if mom[0] != -np.inf:
                        nu_momentum += mom
                        valid_mom_count += 1
                
                if valid_mom_count > 0:
                    interaction.nu_momentum = nu_momentum
                else:
                    interaction.nu_momentum = None
