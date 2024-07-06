import xml.etree.ElementTree as ET
import pandas as pd
import os
from constants import LABELLED_DATA, OBTAINED_DATA


def extract_aspects(sentence):
    aspects = []
    opinions = sentence.find('Opinions')
    if opinions:
        for opinion in opinions.findall('Opinion'):
            if 'target' in opinion.attrib:
                aspects.append(opinion.attrib['target'])
            elif 'implicitTarget' in opinion.attrib:
                aspects.append(opinion.attrib['implicitTarget'])
    return aspects


def parse_annotated_reviews_for_AE(input_file_path, output_file_path):
    """
    Parse an annotated xml file for aspects extraction task
    :param input_file_path: path to the initial file
    :param output_file_path: path to the parsed file
    :return:
    """
    tree = ET.parse(input_file_path)
    root = tree.getroot()
    data = []

    for review in root.findall('Review'):
        full_text = ""
        labels = []
        for sentence in review.find('sentences').findall('sentence'):
            text = sentence.find('text').text
            full_text += text + " "
            opinions = sentence.find('Opinions')
            sentence_labels = ["O"] * len(text.split())  # Initialize all words with 'O' for outside any aspect
            if opinions is not None:
                for opinion in opinions.findall('Opinion'):
                    if 'target' in opinion.attrib:
                        target = opinion.attrib['target']
                    elif 'implicitTarget' in opinion.attrib:
                        target = opinion.attrib['implicitTarget']
                    else:
                        continue  # Skip if neither target nor implicitTarget is available

                    category = opinion.attrib['category']
                    target_tokens = target.split()
                    text_tokens = text.split()
                    target_start_idx = -1
                    for i in range(len(text_tokens) - len(target_tokens) + 1):
                        if text_tokens[i:i + len(target_tokens)] == target_tokens:
                            target_start_idx = i
                            break
                    if target_start_idx != -1:
                        for i in range(len(target_tokens)):
                            if i == 0:
                                sentence_labels[target_start_idx + i] = f"B-{category}"
                            else:
                                sentence_labels[target_start_idx + i] = f"I-{category}"
            labels.extend(sentence_labels)

        data.append((full_text.strip(), labels))

    df = pd.DataFrame(data, columns=['text', 'labels'])
    df.to_pickle(output_file_path)
    return df


xml_path = '/mnt/data/absa_annotated_reviews.xml'
df = parse_annotated_reviews_for_AE(os.path.join(LABELLED_DATA, 'absa_annotated_reviews.xml'),
                                    os.path.join(OBTAINED_DATA, 'absa_aspects_and_categories.pkl'))
print(df.head())
