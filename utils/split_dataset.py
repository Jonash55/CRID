import argparse
import splitfolders

from typing import Tuple


def get_args(arg=None):
    parser = argparse.ArgumentParser(
        description="""
        Specify input path to dataset,
        
        Specify output path where to save files
        If no output path given output - /src/output
        
        Specify ratio of split as tuple (.0, .0, [.0])
        If 2 given, output: train / val
        If 3 given, output: train / val / test
        """
    )
    parser.add_argument("-src", type=tuple, help="path to dataset for splitting")
    parser.add_argument("-out", type=str, help="path where to save output")
    parser.add_argument("-ratio", type=str, help="percentage split as tuple")

    args = parser.parse_args(arg)

    if not args.src:
        print("No source path given - aborting!")
        exit(1)
    elif not args.ratio:
        print("No ratio given - aborting!")
        exit(1)
    elif not args.out:
        print(f"No out path given - saving output to {args.src}/output")
        args.out = f"{args.src}/output"
    return args


def split_data(path_to_data: str, path_to_output: str, ratio: Tuple, seed=1337):
    """
    Split dataset using splitfolders package into three (or two) folders: train, val, test (or train, val)
    """
    splitfolders.ratio(path_to_data, output=path_to_output, seed=seed, ratio=ratio)
    print("Split successful!")


def main():
    args = get_args()
    split_data(args.src, args.out, args.ratio)


if __name__ == "__main__":
    main()
