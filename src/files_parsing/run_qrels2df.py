from parse_functions import parse_qrels
from config.config import *


def main():
    input_file = os.path.join(DATA_PATH, 'inex11sbs.qrels')
    output_pkl_file = os.path.join(OBTAINED_DATA, 'qrels2011.pkl')

    df = parse_qrels(input_file)
    df.to_pickle(output_pkl_file)


if __name__ == '__main__':
    main()
