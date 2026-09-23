"""Export Stage 6 summaries and ignored budget/ID comparison figures."""
import argparse
from pathlib import Path
from stage6_control.reporting import report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/stage6'))
    parser.add_argument('--public',type=Path,default=Path('docs/results/stage6_summary.json'))
    args=parser.parse_args();report(args.run,args.public)
