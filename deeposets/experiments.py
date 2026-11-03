"""High-level experiment controller for the standard paper workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
import optax

from deeposets.configs import one_dim_experiment_config, two_dim_experiment_config
from deeposets.data import describe_dataset, get_1d_data_sampler, get_2d_data_sampler
from deeposets.datasets import generate_burgers1d_dataset, generate_poisson1d_dataset, generate_poisson2d_dataset
from deeposets.evaluation import evaluate_context_sweep
from deeposets.models import DeepOSetsModel, n_param
from deeposets.training import mse_loss, normalized_mse_loss, save_model, train_model


@dataclass(frozen=True)
class ProblemSpec:
    """Metadata that defines paths and defaults for one supported problem."""

    name: str
    display_name: str
    dimension: str
    data_subdir: Path
    output_subdir: Path
    train_dataset_name: str
    test_dataset_name: str
    final_model_name: str
    checkpoint_name: str
    losses_name: str
    evaluation_plot_name: str
    prediction_plot_name: str
    generator: object
    generator_defaults: dict
    config_factory: object
    loss_fn: object


PROBLEM_SPECS = {
    "poisson1d": ProblemSpec(
        name="poisson1d",
        display_name="1D Poisson",
        dimension="1d",
        data_subdir=Path("one_dimension/poisson"),
        output_subdir=Path("one_dimension/poisson"),
        train_dataset_name="poisson1d_dataset_train.npz",
        test_dataset_name="poisson1d_dataset_test.npz",
        final_model_name="deeposets_model_poisson.eqx",
        checkpoint_name="deeposets_checkpoint_poisson.eqx",
        losses_name="losses_poisson.npy",
        evaluation_plot_name="poisson_context_error.png",
        prediction_plot_name="poisson1d_prediction.png",
        generator=generate_poisson1d_dataset,
        generator_defaults={"N": 50, "n_operators": 200_000, "chunk_size": 100, "n_pairs": 40, "seed": 0},
        config_factory=one_dim_experiment_config,
        loss_fn=mse_loss,
    ),
    "burgers1d": ProblemSpec(
        name="burgers1d",
        display_name="1D Burgers",
        dimension="1d",
        data_subdir=Path("one_dimension/burgers"),
        output_subdir=Path("one_dimension/burgers"),
        train_dataset_name="burgers1d_dataset_train.npz",
        test_dataset_name="burgers1d_dataset_test.npz",
        final_model_name="deeposets_model_burgers.eqx",
        checkpoint_name="deeposets_checkpoint_burgers.eqx",
        losses_name="losses_burgers.npy",
        evaluation_plot_name="burgers_context_error.png",
        prediction_plot_name="burgers1d_prediction.png",
        generator=generate_burgers1d_dataset,
        generator_defaults={"N": 100, "M": 1000, "n_operators": 200_000, "chunk_size": 100, "n_pairs": 40, "seed": 0},
        config_factory=one_dim_experiment_config,
        loss_fn=mse_loss,
    ),
    "poisson2d": ProblemSpec(
        name="poisson2d",
        display_name="2D Poisson",
        dimension="2d",
        data_subdir=Path("two_dimension/poisson"),
        output_subdir=Path("two_dimension/poisson"),
        train_dataset_name="poisson2d_dataset_train.npz",
        test_dataset_name="poisson2d_dataset_test.npz",
        final_model_name="deeposets_model_poisson2d.eqx",
        checkpoint_name="deeposets_checkpoint_poisson2d.eqx",
        losses_name="losses_poisson2d.npy",
        evaluation_plot_name="poisson2d_context_error.png",
        prediction_plot_name="poisson2d_prediction.png",
        generator=generate_poisson2d_dataset,
        generator_defaults={"N": 32, "n_operators": 50_000, "chunk_size": 500, "n_pairs": 40, "seed": 0},
        config_factory=two_dim_experiment_config,
        loss_fn=normalized_mse_loss,
    ),
}

PROBLEM_ALIASES = {
    "poisson": "poisson1d",
    "1d-poisson": "poisson1d",
    "poisson-1d": "poisson1d",
    "burgers": "burgers1d",
    "1d-burgers": "burgers1d",
    "burgers-1d": "burgers1d",
    "2d-poisson": "poisson2d",
    "poisson-2d": "poisson2d",
}


def normalize_problem_name(problem):
    """Accept short command-line aliases while keeping one canonical name."""
    key = problem.lower()
    return PROBLEM_ALIASES.get(key, key)


def get_problem_spec(problem):
    """Return the metadata for one supported experiment."""
    name = normalize_problem_name(problem)
    if name not in PROBLEM_SPECS:
        supported = ", ".join(sorted(PROBLEM_SPECS))
        raise ValueError(f"Unsupported problem '{problem}'. Supported problems: {supported}")
    return PROBLEM_SPECS[name]


class DeepOSetExperiment:
    """Reusable controller for data generation, training, evaluation, and plotting."""

    def __init__(self, problem, data_root="data", output_root="outputs"):
        self.spec = get_problem_spec(problem)
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)

    @property
    def data_dir(self):
        return self.data_root / self.spec.data_subdir

    @property
    def output_dir(self):
        return self.output_root / self.spec.output_subdir

    @property
    def train_dataset_path(self):
        return self.data_dir / self.spec.train_dataset_name

    @property
    def test_dataset_path(self):
        return self.data_dir / self.spec.test_dataset_name

    @property
    def final_model_path(self):
        return self.output_dir / self.spec.final_model_name

    @property
    def checkpoint_path(self):
        return self.output_dir / self.spec.checkpoint_name

    @property
    def losses_path(self):
        return self.output_dir / self.spec.losses_name

    @property
    def evaluation_plot_path(self):
        return self.output_root / "evaluation" / self.spec.evaluation_plot_name

    @property
    def evaluation_metrics_path(self):
        stem = Path(self.spec.evaluation_plot_name).stem
        return self.output_root / "evaluation" / f"{stem}_metrics.npz"

    @property
    def prediction_plot_path(self):
        return self.output_root / "figures" / self.spec.prediction_plot_name

    def config(self):
        return self.spec.config_factory()

    def paths(self):
        """Return all default paths as strings for notebooks and README examples."""
        return {
            "data_dir": str(self.data_dir),
            "train_dataset": str(self.train_dataset_path),
            "test_dataset": str(self.test_dataset_path),
            "output_dir": str(self.output_dir),
            "final_model": str(self.final_model_path),
            "checkpoint": str(self.checkpoint_path),
            "losses": str(self.losses_path),
            "evaluation_plot": str(self.evaluation_plot_path),
            "evaluation_metrics": str(self.evaluation_metrics_path),
            "prediction_plot": str(self.prediction_plot_path),
        }

    def data_exists(self):
        return self.train_dataset_path.exists() and self.test_dataset_path.exists()

    def model_exists(self):
        return self.final_model_path.exists()

    def describe_data(self, split="train", dataset_path=None):
        path = Path(dataset_path) if dataset_path else self.dataset_path(split)
        return describe_dataset(path)

    def dataset_path(self, split):
        if split == "train":
            return self.train_dataset_path
        if split == "test":
            return self.test_dataset_path
        raise ValueError("split must be 'train' or 'test'")

    def generate_data(self, force=False, **overrides):
        """Generate train/test datasets, or reuse them if they already exist."""
        if self.data_exists() and not force:
            print(f"Using existing datasets in {self.data_dir}")
            return self.train_dataset_path, self.test_dataset_path

        kwargs = dict(self.spec.generator_defaults)
        kwargs.update({key: value for key, value in overrides.items() if value is not None})
        self.spec.generator(save_dir=self.data_dir, **kwargs)
        return self.train_dataset_path, self.test_dataset_path

    def load_sampler(self, split="train", n_eqns=None, num_ucpair=None, dataset_path=None):
        """Load a batch sampler for existing generated data."""
        config = self.config()
        path = Path(dataset_path) if dataset_path else self.dataset_path(split)
        n_eqns = config["data_param"]["n_eqns"] if n_eqns is None else n_eqns
        num_ucpair = config["data_param"]["num_ucpair"] if num_ucpair is None else num_ucpair

        if self.spec.dimension == "1d":
            return get_1d_data_sampler(path, n_eqns, num_ucpair)
        return get_2d_data_sampler(path, n_eqns, num_ucpair)

    def build_model(self, split="train", n_eqns=None, num_ucpair=None, dataset_path=None, seed=None):
        """Create an untrained model template whose shapes match a dataset."""
        config = self.config()
        seed = config["seed"] if seed is None else seed
        sampler = self.load_sampler(split=split, n_eqns=n_eqns, num_ucpair=num_ucpair, dataset_path=dataset_path)
        batch = sampler(jr.PRNGKey(seed))
        return DeepOSetsModel.from_data(batch, **config["model_param"], key=jr.PRNGKey(seed + 1))

    def load_model(self, model_path=None, split="test", n_eqns=None, num_ucpair=None, dataset_path=None):
        """Load a saved Equinox model without retraining."""
        path = Path(model_path) if model_path else self.final_model_path
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        template = self.build_model(split=split, n_eqns=n_eqns, num_ucpair=num_ucpair, dataset_path=dataset_path, seed=0)
        return eqx.tree_deserialise_leaves(path, template)

    def train(self, dataset_path=None, steps=None, save_every=5000, reuse_existing=False):
        """Train the model, or return the existing model path when requested."""
        dataset_path = Path(dataset_path) if dataset_path else self.train_dataset_path
        if reuse_existing and self.final_model_path.exists():
            print(f"Using existing trained model: {self.final_model_path}")
            return self.final_model_path

        config = self.config()
        if steps is not None:
            config["training_param"]["n_step"] = steps

        self.output_dir.mkdir(parents=True, exist_ok=True)
        keys = jr.split(jr.PRNGKey(config["seed"]), 10)
        sampler = self.load_sampler(
            split="train",
            dataset_path=dataset_path,
            n_eqns=config["data_param"]["n_eqns"],
            num_ucpair=config["data_param"]["num_ucpair"],
        )
        batch = sampler(keys[0])
        model = DeepOSetsModel.from_data(batch, **config["model_param"], key=keys[1])
        print(f"Model parameters: {n_param(model):,}")

        lr = optax.cosine_decay_schedule(
            config["training_param"]["init_lr"],
            config["training_param"]["n_step"],
            alpha=1e-6,
        )
        optimizer = optax.adam(lr)
        model, losses = train_model(
            model,
            sampler,
            optimizer,
            config["training_param"]["n_step"],
            key=keys[2],
            loss_fn=self.spec.loss_fn,
            checkpoint_path=self.checkpoint_path,
            save_every=save_every,
        )
        save_model(self.final_model_path, model)
        np.save(self.losses_path, losses)
        print(f"Saved final model and losses to {self.output_dir}")
        return self.final_model_path

    def evaluate(
        self,
        dataset_path=None,
        model_path=None,
        min_ucpair=3,
        max_ucpair=None,
        n_trials=100,
        n_eqns=20,
        seed=42,
        save_plot=None,
        save_metrics=None,
    ):
        """Evaluate an existing trained model across context sizes."""
        config = self.config()
        max_ucpair = config["data_param"]["num_ucpair"] if max_ucpair is None else max_ucpair
        dataset_path = Path(dataset_path) if dataset_path else self.test_dataset_path
        save_plot = Path(save_plot) if save_plot else self.evaluation_plot_path
        save_metrics = Path(save_metrics) if save_metrics else self.evaluation_metrics_path

        model = self.load_model(
            model_path=model_path,
            split="test",
            dataset_path=dataset_path,
            n_eqns=n_eqns,
            num_ucpair=max(min_ucpair, 3),
        )

        def sampler_factory(num_ucpair):
            return self.load_sampler(split="test", n_eqns=n_eqns, num_ucpair=num_ucpair, dataset_path=dataset_path)

        context_sizes = range(min_ucpair, max_ucpair + 1)
        results = evaluate_context_sweep(model, sampler_factory, context_sizes, n_trials=n_trials, seed=seed)
        for num_ucpair, rel_l2, mse in zip(results["context_sizes"], results["relative_l2"], results["mse"]):
            print(f"ucpair={int(num_ucpair):2d}: rel_L2={rel_l2:.4f}, MSE={mse:.2e}")

        from deeposets.plotting import plot_context_error

        plot_context_error(
            results["context_sizes"],
            results["relative_l2"],
            f"Accuracy vs Context Size - {self.spec.display_name}",
            save_plot,
        )
        save_metrics.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_metrics, **results)
        print(f"Saved plot to {save_plot}")
        print(f"Saved metrics to {save_metrics}")
        return results

    def plot_prediction(
        self,
        dataset_path=None,
        model_path=None,
        num_ucpair=5,
        n_eqns=4,
        operator_index=0,
        seed=22,
        save_plot=None,
    ):
        """Plot one sampled prediction using an existing model."""
        dataset_path = Path(dataset_path) if dataset_path else self.test_dataset_path
        save_plot = Path(save_plot) if save_plot else self.prediction_plot_path

        sampler = self.load_sampler(split="test", n_eqns=n_eqns, num_ucpair=num_ucpair, dataset_path=dataset_path)
        data = sampler(jr.PRNGKey(seed))
        model = self.load_model(
            model_path=model_path,
            split="test",
            n_eqns=n_eqns,
            num_ucpair=num_ucpair,
            dataset_path=dataset_path,
        )

        if self.spec.dimension == "1d":
            from deeposets.plotting import plot_1d_prediction

            problem = "burgers" if self.spec.name == "burgers1d" else "poisson"
            plot_1d_prediction(model, data, problem, save_plot, operator_index=operator_index)
        else:
            from deeposets.plotting import plot_2d_prediction

            plot_2d_prediction(model, data, save_plot, operator_index=operator_index)

        print(f"Saved plot to {save_plot}")
        return save_plot
