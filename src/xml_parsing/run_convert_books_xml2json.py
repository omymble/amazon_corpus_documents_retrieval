from parse_functions import parse_book_xml, convert_xml_to_json
from config.config import *


def main():
    input_directory = BOOKS_DATA
    output_directory = os.path.join(OBTAINED_DATA, 'old_collection/')

    convert_xml_to_json(input_directory, output_directory)


if __name__ == '__main__':
    main()
