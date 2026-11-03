"""Run standard DeepOSets workflows from one command-line interface."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

from deeposets.experiments import DeepOSetExperiment, PROBLEM_SPECS

jax.config.update("jax_enable_x64", False)


def selected_stages(stage):
    """Expand the `all` stage into the ordered workflow stages."""
    if stage == "all":
        return ("data", "train", "evaluate", "plot")
    return (stage,)


def main():
    """Parse CLI arguments and dispatch the requested experiment stage."""
    parser = argparse.ArgumentParser(
        description="Run one modular DeepOSets workflow stage: data, train, evaluate, plot, or all."
    )
    parser.add_argument("--problem", choices=sorted(PROBLEM_SPECS), default="poisson1d")
    parser.add_argument("--stage", choices=["data", "train", "evaluate", "plot", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--force", action="store_true", help="Regenerate data and retrain even when outputs already exist.")

    parser.add_argument("--N", type=int, default=None, help="Override the spatial grid size for data generation.")
    parser.add_argument("--M", type=int, default=None, help="Override the Burgers time-step count for data generation.")
    parser.add_argument("--n-operators", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--n-pairs", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)

    parser.add_argument("--steps", type=int, default=None, help="Override the number of training steps.")
    parser.add_argument("--save-every", type=int, default=5000)

    parser.add_argument("--min-ucpair", type=int, default=3)
    parser.add_argument("--max-ucpair", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--eval-n-eqns", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=42)

    parser.add_argument("--num-ucpair", type=int, default=5)
    parser.add_argument("--plot-n-eqns", type=int, default=4)
    parser.add_argument("--operator-index", type=int, default=0)
    parser.add_argument("--plot-seed", type=int, default=22)
    args = parser.parse_args()

    experiment = DeepOSetExperiment(args.problem, data_root=args.data_root, output_root=args.output_root)
    stages = selected_stages(args.stage)

    if "data" in stages:
        generation_overrides = {
            "N": args.N,
            "M": args.M,
            "n_operators": args.n_operators,
            "chunk_size": args.chunk_size,
            "n_pairs": args.n_pairs,
            "seed": args.data_seed,
        }
        experiment.generate_data(force=args.force, **generation_overrides)

    if "train" in stages:
        experiment.train(steps=args.steps, save_every=args.save_every, reuse_existing=not args.force)

    if "evaluate" in stages:
        experiment.evaluate(
            min_ucpair=args.min_ucpair,
            max_ucpair=args.max_ucpair,
            n_trials=args.n_trials,
            n_eqns=args.eval_n_eqns,
            seed=args.eval_seed,
        )

    if "plot" in stages:
        experiment.plot_prediction(
            num_ucpair=args.num_ucpair,
            n_eqns=args.plot_n_eqns,
            operator_index=args.operator_index,
            seed=args.plot_seed,
        )

    print("Workflow complete. Paths:")
    for label, path in experiment.paths().items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
