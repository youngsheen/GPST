import argparse
import glob
import os
import random
import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument(
        "--name", type=str
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    rand = random.Random(args.seed)

    valid_f = (
        open(os.path.join(args.dest, f"{args.name}_valid.tsv"), "w")
        if args.valid_percent > 0
        else None
    )

    with open(os.path.join(args.dest, f"{args.name}.tsv"), "w") as train_f:

        for fname in tqdm.tqdm(glob.iglob(search_path, recursive=True)):
            file_path = os.path.realpath(fname)
            
            if valid_f is None:
                dest = train_f
            else:
                dest = train_f if rand.random() > args.valid_percent else valid_f
                
            print(file_path, file=dest)
            
    if valid_f is not None:
        valid_f.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
