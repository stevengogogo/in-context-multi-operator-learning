"""Command-line entry point for generating the 1D Poisson dataset."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deeposets.datasets import generate_poisson1d_dataset


def main():
    """Parse CLI arguments and generate train/test `.npz` files."""
    parser = argparse.ArgumentParser(description="Generate the 1D Poisson dataset.")
    parser.add_argument("--save-dir", default="data/one_dimension/poisson")
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--n-operators", type=int, default=200_000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--n-pairs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse-existing", action="store_true", help="Skip generation when train/test files already exist.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    train_path = save_dir / "poisson1d_dataset_train.npz"
    test_path = save_dir / "poisson1d_dataset_test.npz"
    if args.reuse_existing and train_path.exists() and test_path.exists():
        print(f"Using existing 1D Poisson datasets in {save_dir}")
        return

    generate_poisson1d_dataset(
        save_dir=save_dir,
        N=args.N,
        n_operators=args.n_operators,
        chunk_size=args.chunk_size,
        n_pairs=args.n_pairs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
