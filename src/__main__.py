from corpus import Loader, Creator
import pandas as pd


def main():
    # loader = Loader('../data/documents')
    # loader.save_tokens('../data/tokens.csv')
    df = pd.read_csv('../data/tokens.csv')
    creator = Creator(tokens=df)
    creator.save_interleaves('../data/interleaved.csv')


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
