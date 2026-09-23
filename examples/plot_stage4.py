"""Render saved Stage 4 measurements without rerunning model evaluation."""
from stage4_evaluation.plotting import make_figures

if __name__=='__main__':
    for path in make_figures():
        print(path)
