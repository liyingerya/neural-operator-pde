"""Generate separate OOD archives, or the matched resolution pair."""
import argparse
from pathlib import Path
from stage4_evaluation.ood import generate_evaluation_data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('data/stage4'))
    parser.add_argument('--resolution',action='store_true')
    args=parser.parse_args()
    generate_evaluation_data(args.output,args.resolution)


if __name__=='__main__':
    main()
