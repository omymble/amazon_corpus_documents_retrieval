from parse_functions import (extract_aspects_categories_polarities, extract_aspects_per_sentences,
                             extract_every_aspect, absa_xml_to_setfit_df)
from config.config import *
import argparse

labelled_absa_file = os.path.join(LABELLED_DATA, 'absa_annotated_reviews.xml')


def main():
    output_file_path = os.path.join(OBTAINED_DATA, 'setfit_categories.pkl')
    df = absa_xml_to_setfit_df(labelled_absa_file)
    df.to_pickle(output_file_path)
    print(f"DataFrame saved to {output_file_path}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Run ABSA XML to DataFrame conversion.")
    # parser.add_argument('model_name', type=str, help='The name of the model to use.')
    # args = parser.parse_args()
    main()
