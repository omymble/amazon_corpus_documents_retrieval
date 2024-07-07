import xml.etree.ElementTree as ET
import pandas as pd


def extract_aspects_categories_polarities(input_file_path):
    """
    saves pandas dataframe with new attributes extracted
    from labelled xml file
    :param input_file_path: initial file
    :return: pandas dataframe with 'text', 'aspects', 'categories', 'polarities' attributes
    """
    tree = ET.parse(input_file_path)
    root = tree.getroot()

    texts = []
    targets = []
    categories = []
    polarities = []

    for review in root.findall('Review'):
        review_text = []
        review_targets = []
        review_categories = []
        review_polarities = []

        for sentence in review.find('sentences').findall('sentence'):
            # Extract the text and append it to the review_text list
            text = sentence.find('text').text
            review_text.append(text)

            # Extract opinions if they exist
            opinions = sentence.find('Opinions')
            if opinions is not None:
                for opinion in opinions.findall('Opinion'):
                    # Extract the target, category, and polarity
                    target = opinion.attrib.get('target')
                    if target:  # Only include explicit targets
                        category = opinion.attrib.get('category')
                        polarity = opinion.attrib.get('polarity')

                        # Append to the lists
                        review_targets.append(target)
                        review_categories.append(category)
                        review_polarities.append(polarity)

            # Join the review_text list to form a single string
        full_text = ' '.join(review_text)

        # Append the extracted data to the main lists
        if review_targets:
            texts.append(full_text)
            targets.append(review_targets)
            categories.append(review_categories)
            polarities.append(review_polarities)

    # Create a DataFrame
    df = pd.DataFrame({
        'text': texts,
        'targets': targets,
        'categories': categories,
        'polarities': polarities
    })

    return df


def parse_user_request(input_file_path):
    return
