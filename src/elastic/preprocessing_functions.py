from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import Document
import os
import json


def read_local_json_files(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    documents = []

    for file in json_files:
        with open(os.path.join(directory, file), 'r') as f:
            book_data = json.load(f)

            # Build the text field from specific JSON elements
            text_field = json.dumps({
                "title": book_data.get("title"),
                "reviews": book_data.get("reviews", []),
                "editorialreviews": book_data.get("editorialreviews", [])
            })

            # Building Document required by LlamaIndex
            doc = Document(
                text=text_field,
                metadata={
                    "isbn": book_data.get("isbn"),
                    "ean": book_data.get("ean"),
                    "binding": book_data.get("binding"),
                    "label": book_data.get("label"),
                    "listprice": book_data.get("listprice"),
                    "manufacturer": book_data.get("manufacturer"),
                    "publisher": book_data.get("publisher"),
                    "readinglevel": book_data.get("readinglevel"),
                    "releasedate": book_data.get("releasedate"),
                    "publicationdate": book_data.get("publicationdate"),
                    "studio": book_data.get("studio"),
                    "edition": book_data.get("edition"),
                    "dewey": book_data.get("dewey"),
                    "numberofpages": book_data.get("numberofpages"),
                    "dimensions": book_data.get("dimensions"),
                    "creators": book_data.get("creators"),
                    "blurbers": book_data.get("blurbers"),
                    "dedications": book_data.get("dedications"),
                    "epigraphs": book_data.get("epigraphs"),
                    "firstwords": book_data.get("firstwords"),
                    "lastwords": book_data.get("lastwords"),
                    "quotations": book_data.get("quotations"),
                    "series": book_data.get("series"),
                    "awards": book_data.get("awards"),
                    "characters": book_data.get("characters"),
                    "places": book_data.get("places"),
                    "subjects": book_data.get("subjects"),
                    "tags": book_data.get("tags"),
                    "similarproducts": book_data.get("similarproducts"),
                    "browseNodes": book_data.get("browseNodes")
                }
            )
            documents.append(doc)

    return documents
