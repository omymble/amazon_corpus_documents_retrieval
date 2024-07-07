from parse_functions import extract_aspects_categories_polarities
from config.config import *


def main():
    input_file_path = os.path.join(LABELLED_DATA, 'absa_annotated_reviews.xml')
    output_file_path = os.path.join(OBTAINED_DATA, 'absa_aspects_categories_polarities.pkl')

    df = extract_aspects_categories_polarities(input_file_path)
    df.to_pickle(output_file_path)
    print(f"DataFrame saved to {output_file_path}")


if __name__ == "__main__":
    main()