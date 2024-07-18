import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
from config.config import *
import json
import re
import html
from io import StringIO


def clean_narrative(narrative):
    """
    Clean request texts in requests2011.xml
    """
    narrative = html.unescape(narrative)
    narrative = re.sub(r'<br\s*/?>', ' ', narrative)
    narrative = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', narrative)
    return narrative.strip()


def clean_xml_content(xml_content):
    """
    Clean XML content to escape special characters
    """
    xml_content = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', xml_content)
    return xml_content


def parse_topic(topic):
    """
    Extract information from <topic> tag in requests2011.xml
    """
    topic_data = {}
    topic_data['id'] = int(topic.get('id'))

    title = html.unescape(topic.find('title').text.strip())
    narrative = html.unescape(topic.find('narrative').text.strip())
    narrative_cleaned = clean_narrative(narrative)
    if not re.match(r'.*[.!?]$', title):
        title += '.'
    topic_data['request'] = f"{title} {narrative_cleaned}"

    topic_data['group'] = html.unescape(topic.find('group').text.strip())

    types = topic.find('types')
    topic_data['types'] = [type_elem.text for type_elem in types.findall('type')]

    genres = topic.find('genres')
    genre_text = ' '.join([html.unescape(genre.text) for genre in genres.findall('genre')])
    genre_list = re.split(r'[.\-]', genre_text)
    topic_data['genres'] = [genre.strip() for genre in genre_list if genre.strip()]

    specificity = topic.find('specificity').text.strip()
    topic_data['specificity'] = specificity

    similar_isbn = []
    similar_author = []
    for similar in topic.findall('similar'):
        for work in similar.findall('work'):
            isbn = work.find('isbn')
            if isbn is not None:
                similar_isbn.append(isbn.text.strip())
        for author in similar.findall('author'):
            similar_author.append(author.text.strip())
    topic_data['similar_isbn'] = similar_isbn
    topic_data['similar_author'] = similar_author

    dissimilar_isbn = []
    dissimilar_author = []
    for dissimilar in topic.findall('dissimilar'):
        for work in dissimilar.findall('work'):
            isbn = work.find('isbn')
            if isbn is not None:
                dissimilar_isbn.append(isbn.text.strip())
        for author in dissimilar.findall('author'):
            dissimilar_author.append(author.text.strip())
    topic_data['dissimilar_isbn'] = dissimilar_isbn
    topic_data['dissimilar_author'] = dissimilar_author

    return topic_data


def parse_requests_xml_to_df(input_xml_file):
    """
    Creates pandas df from nested requests2011.xml
    """

    with open(input_xml_file, 'r', encoding='utf-8') as file:
        xml_content = file.read()

    # Clean XML content
    xml_content = clean_xml_content(xml_content)

    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()

    topics_data = []
    for topic in root.findall('topic'):
        topics_data.append(parse_topic(topic))

    df = pd.DataFrame(topics_data)
    df.set_index('id', inplace=True)

    return df


def parse_qrels(input_file_path) -> pd.DataFrame:
    columns = ['topic', 'ignore1', 'doc_id', 'relevance']
    dtype = {'topic': int, 'ignore1': int, 'doc_id': int, 'relevance': int}
    df = pd.read_csv(input_file_path, sep=' ', header=None, names=columns, dtype=dtype)
    df.drop(columns=['ignore1'], inplace=True)
    return df


def extract_aspects_per_sentences(input_file_path):
    tree = ET.parse(input_file_path)
    root = tree.getroot()
    data = []

    for review in root.findall('Review'):
        rid = review.get('rid')
        review_sentences = review.find('sentences').findall('sentence')
        review_texts = []

        for sentence in review_sentences:
            text = sentence.find('text').text
            if review_texts and not review_texts[-1].strip().endswith(('.', '!', '?')):
                review_texts[-1] += '. ' + text
            else:
                review_texts.append(text)

        review_text = " ".join(review_texts)

        for sentence in review_sentences:
            sentence_id = sentence.get('id')
            text = sentence.find('text').text
            opinions = sentence.find('Opinions')
            targets = set()
            if opinions:
                for opinion in opinions.findall('Opinion'):
                    target = opinion.get('target')
                    if target:
                        targets.add(target)

            data.append({
                'review_id': rid,
                'review_text': review_text,
                'sentence_id': sentence_id,
                'sentence_text': text,
                'targets': list(targets)
            })

    df = pd.DataFrame(data)
    df['num_targets'] = df['targets'].apply(lambda x: len(x))
    return df


def extract_aspects_categories_polarities(input_file_path):
    """
    saves pandas dataframe with new attributes
    extracted from labelled xml file. Extracts information on the text level.
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


def split_requests_xml(file_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    tree = ET.parse(file_path)
    root = tree.getroot()

    for topic in root.findall('topic'):
        topicid = topic.find('topicid').text
        topic_tree = ET.ElementTree(topic)
        output_file_path = os.path.join(output_folder, f'{topicid}.xml')
        topic_tree.write(output_file_path, encoding='utf-8', xml_declaration=True)

    print(f"XML file split into {len(root.findall('topic'))} files in {output_folder}")


def parse_date(date_str):
    """Keep only the year for books data"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").year
    except ValueError:
        return None


def parse_book_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    book = {
        "isbn": root.findtext("isbn"),
        "title": root.findtext("title"),
        "ean": root.findtext("ean"),
        "binding": root.findtext("binding"),
        "label": root.findtext("label"),
        "listprice": root.findtext("listprice"),
        "manufacturer": root.findtext("manufacturer"),
        "publisher": root.findtext("publisher"),
        "readinglevel": root.findtext("readinglevel"),
        "releasedate": parse_date(root.findtext("releasedate")),
        "publicationdate": parse_date(root.findtext("publicationdate")),
        "studio": root.findtext("studio"),
        "edition": root.findtext("edition"),
        "dewey": root.findtext("dewey"),
        "numberofpages": root.findtext("numberofpages"),
        "dimensions": {
            "height": root.find("dimensions/height").text if root.find("dimensions/height") is not None else None,
            "width": root.find("dimensions/width").text if root.find("dimensions/width") is not None else None,
            "length": root.find("dimensions/length").text if root.find("dimensions/length") is not None else None,
            "weight": root.find("dimensions/weight").text if root.find("dimensions/weight") is not None else None,
        },
        "reviews": [],
        "editorialreviews": [],
        "images": [],
        "creators": [],
        "blurbers": [blurber.text for blurber in root.findall("blurbers/blurber")],
        "dedications": [dedication.text for dedication in root.findall("dedications/dedication")],
        "epigraphs": [epigraph.text for epigraph in root.findall("epigraphs/epigraph")],
        "firstwords": [firstword.text for firstword in root.findall("firstwords/firstwordsitem")],
        "lastwords": [lastword.text for lastword in root.findall("lastwords/lastwordsitem")],
        "quotations": [quotation.text for quotation in root.findall("quotations/quotation")],
        "series": [seriesitem.text for seriesitem in root.findall("series/seriesitem")],
        "awards": [award.text for award in root.findall("awards/award")],
        "characters": [character.text for character in root.findall("characters/character")],
        "places": [place.text for place in root.findall("places/place")],
        "subjects": [subject.text for subject in root.findall("subjects/subject")],
        "tags": [{"tag": tag.text, "count": tag.get("count")} for tag in root.findall("tags/tag")],
        "similarproducts": [similarproduct.text for similarproduct in root.findall("similarproducts/similarproduct")],
        "browseNodes": [{"browseNode": browseNode.text, "id": browseNode.get("id")} for browseNode in
                        root.findall("browseNodes/browseNode")]
    }

    for review in root.findall("reviews/review"):
        book["reviews"].append({
            "authorid": review.findtext("authorid"),
            "date": parse_date(review.findtext("date")),
            "summary": review.findtext("summary"),
            "content": review.findtext("content"),
            "rating": review.findtext("rating"),
            "totalvotes": review.findtext("totalvotes"),
            "helpfulvotes": review.findtext("helpfulvotes"),
        })

    for editorialreview in root.findall("editorialreviews/editorialreview"):
        book["editorialreviews"].append({
            "source": editorialreview.findtext("source"),
            "content": editorialreview.findtext("content"),
        })

    for image in root.findall("images/image"):
        book["images"].append({
            "url": image.findtext("url"),
            "height": image.findtext("height"),
            "width": image.findtext("width"),
            "imageCategories": [category.text for category in image.findall("imageCategories/imagecategory")]
        })

    for creator in root.findall("creators/creator"):
        book["creators"].append({
            "name": creator.findtext("name"),
            "role": creator.findtext("role"),
        })

    return book


def convert_xml_to_json(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root_dir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_file = os.path.join(root_dir, file)
                book_data = parse_book_xml(xml_file)
                json_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.json")
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(book_data, f, ensure_ascii=False, indent=4)
