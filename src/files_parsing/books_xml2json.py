import xml.etree.ElementTree as ET
from config.config import *
import json
from parse_functions import parse_date


dewey_dict = {
    "000": "Generalities",
    "100": "Philosophy & Psychology",
    "200": "Religion",
    "300": "Social Sciences",
    "400": "Language",
    "500": "Science",
    "600": "Technology",
    "700": "Arts & Recreation",
    "800": "Literature",
    "900": "History & Geography"
}


def get_dewey_description(dewey_number):
    if dewey_number is None:
        return None
    key = str(dewey_number).zfill(3)[:1] + "00"
    return dewey_dict.get(key)


def parse_book_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    dewey_number = root.findtext("dewey")
    dewey_description = get_dewey_description(dewey_number)

    book = {
        "isbn": root.findtext("isbn"),
        "title": root.findtext("title"),
        "publisher": root.findtext("publisher"),
        "readinglevel": root.findtext("readinglevel"),
        "publicationdate": parse_date(root.findtext("publicationdate")),
        "edition": root.findtext("edition"),
        "dewey": [dewey_description] if dewey_description else [],
        "reviews": [],
        "editorialreviews": [],
        "authors": [],
        "characters": [character.text for character in root.findall("characters/character")],
        "places": [place.text for place in root.findall("places/place")],
        "subjects": [subject.text for subject in root.findall("subjects/subject")],
        "tags": [tag.text for tag in root.findall("tags/tag")],
        "browseNodes": [browseNode.text for browseNode in root.findall("browseNodes/browseNode")]
    }

    for review in root.findall("reviews/review"):
        summary = review.findtext("summary")
        content = review.findtext("content")
        merged_review = f"{summary} - {content}" if summary and content else summary or content
        if merged_review:
            book["reviews"].append(merged_review)

    for editorialreview in root.findall("editorialreviews/editorialreview"):
        book["editorialreviews"].append(
            editorialreview.findtext("content")
        )

    for creator in root.findall("creators/creator"):
        name = creator.findtext("name")
        if name:
            book["authors"].append(name)

    blurbers = [blurber.text for blurber in root.findall("blurbers/blurber")]
    epigraphs = [epigraph.text for epigraph in root.findall("epigraphs/epigraph")]
    firstwords = [firstwordsitem.text for firstwordsitem in root.findall("firstwords/firstwordsitem")]
    lastwords = [lastwordsitem.text for lastwordsitem in root.findall("lastwords/lastwordsitem")]
    quotations = [quotation.text for quotation in root.findall("quotations/quotation")]

    content = " ".join(blurbers + epigraphs + firstwords + lastwords + quotations).strip()
    if content:
        book["content"] = content

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


def main():
    input_directory = BOOKS_DATA
    output_directory = os.path.join(OBTAINED_DATA, 'books_collection/')

    convert_xml_to_json(input_directory, output_directory)


if __name__ == '__main__':
    main()
