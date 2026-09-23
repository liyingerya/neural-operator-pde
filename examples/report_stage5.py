"""Export public ablation summaries and figures from frozen evaluation outputs."""
import argparse
from pathlib import Path
from stage5_training.reporting import report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/stage5'))
    parser.add_argument('--public',type=Path,default=Path('docs/results/stage5_summary.json'),
                        help='New summary path; existing JSON/CSV exports are never overwritten')
    args=parser.parse_args()
    report(args.run,args.public)
