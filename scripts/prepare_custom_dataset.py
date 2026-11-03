"""Convert user-provided `.npz` arrays into the custom DeepOSets format."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deeposets.data import describe_dataset, prepare_custom_dataset_from_npz


def main():
    """Parse CLI arguments, convert arrays, and print saved dataset metadata."""
    parser = argparse.ArgumentParser(
        description="Convert user-provided .npz arrays into the standard DeepOSets custom dataset format."
    )
    parser.add_argument("--source", type=Path, required=True, help="Path to a user .npz file.")
    parser.add_argument("--save-dir", type=Path, default=Path("data/custom"))
    parser.add_argument("--input-key", default="cond", help="Array key for input/query fields.")
    parser.add_argument("--output-key", default="qoi", help="Array key for output/solution fields.")
    parser.add_argument("--coords-key", default="xs", help="Optional coordinate array key.")
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--prefix", default="custom")
    args = parser.parse_args()

    train_path, test_path = prepare_custom_dataset_from_npz(
        source_path=args.source,
        save_dir=args.save_dir,
        input_key=args.input_key,
        output_key=args.output_key,
        coords_key=args.coords_key,
        train_fraction=args.train_fraction,
        seed=args.seed,
        shuffle=args.shuffle,
        prefix=args.prefix,
    )
    print(f"Saved train dataset: {train_path}")
    print(f"Saved test dataset:  {test_path}")
    print(describe_dataset(train_path))
    print(describe_dataset(test_path))


if __name__ == "__main__":
    main()
