"""Small in-process smoke test for data generation, sampling, and training."""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.random as jr
import optax

from deeposets.configs import one_dim_experiment_config
from deeposets.data import get_1d_data_sampler
from deeposets.datasets import generate_poisson1d_dataset
from deeposets.models import DeepOSetsModel
from deeposets.training import train_model


def main():
    """Generate tiny data, train for one step, and assert a loss was recorded."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        generate_poisson1d_dataset(data_dir, N=8, n_operators=4, chunk_size=2, n_pairs=4)

        config = one_dim_experiment_config()
        config["data_param"]["n_eqns"] = 2
        config["data_param"]["num_ucpair"] = 3
        config["model_param"]["widths_net_deepset"] = [16]
        config["model_param"]["widths_net_branch"] = [16]
        config["model_param"]["widths_net_trunk"] = [16]
        config["model_param"]["n_embed_h"] = 16
        config["model_param"]["n_intermediate"] = 16

        keys = jr.split(jr.PRNGKey(0), 3)
        sampler = get_1d_data_sampler(data_dir / "poisson1d_dataset_train.npz", 2, 3)
        batch = sampler(keys[0])
        model = DeepOSetsModel.from_data(batch, **config["model_param"], key=keys[1])
        optimizer = optax.adam(1e-4)
        _, losses = train_model(model, sampler, optimizer, 1, keys[2])
        assert losses.shape == (1,)
        print("Smoke test passed.")


if __name__ == "__main__":
    main()
