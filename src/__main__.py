from corpus import Loader, Creator
from src.models import W2V
import pandas as pd

from src.visualization import Visualizer


def main():
    model = W2V(path='../models/interleaved')
    vis = Visualizer(model, '../data/tokens.csv')
    vis.visualize_most_frequent(limit=100)


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
