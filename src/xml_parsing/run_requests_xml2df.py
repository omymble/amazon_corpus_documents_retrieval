from parse_functions import parse_requests_xml_to_df
from config.config import *


def main():
    input_xml_file = os.path.join(DATA_PATH, 'requests2011.xml')
    output_pkl_file = os.path.join(OBTAINED_DATA, 'requests2011.pkl')

    df = parse_requests_xml_to_df(input_xml_file)
    df.to_pickle(output_pkl_file)


if __name__ == '__main__':
    main()

