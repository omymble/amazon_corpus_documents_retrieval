from parse_functions import split_requests_xml
from config.config import *


def main():
    input_file = os.path.join(DATA_PATH, 'requests.xml')
    output_folder = os.path.join(SPLIT_REQUESTS)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    split_requests_xml(input_file, output_folder)


if __name__ == '__main__':
    main()

