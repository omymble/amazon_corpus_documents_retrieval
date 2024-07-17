import re
import argparse
import xmltodict
import json
from search import Search

es = Search()


def extract_filters(query):
    filter_regex = r"category:([^\s]+)\s*"
    m = re.search(filter_regex, query)
    if m is None:
        return {}, query  # no filters
    filters = {"term": {"category.keyword": m.group(1)}}
    query = re.sub(filter_regex, "", query).strip()
    return filters, query


def handle_search(index, query, from_=0):
    filters, parsed_query = extract_filters(query)

    if parsed_query:
        search_query = {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": parsed_query,
                        "fields": ["title", "authors.name", "tags.tag"],
                    }
                },
                "filter": filters
            }
        }
    else:
        search_query = {"match_all": {}}

    results = es.search(index=index, query=search_query, from_=from_)

    print("Results:")
    for hit in results["hits"]["hits"]:
        print(hit["_source"])
    print("\nTotal Hits:", results["hits"]["total"]["value"])


def get_document(index, doc_id):
    document = es.retrieve_document(index=index, doc_id=doc_id)
    print(json.dumps(document["_source"], indent=4))


def reindex(index):
    es.reindex(index=index)
    print(f"Index '{index}' re-created.")


def index_json_file(index, json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        doc_id = data.get('isbn', None)
        if doc_id:
            es.index_document(index=index, doc_id=doc_id, document=data)
            print(f"Document {doc_id} indexed successfully.")


def process_xml_request(xml_file):
    with open(xml_file, 'r') as file:
        content = file.read()
        doc = xmltodict.parse(content)
        topic = doc.get("topic")
        if topic:
            title = topic.get("title")
            request = topic.get("request")
            print(f"Title: {title}\nRequest: {request}")
            return title, request
    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search application CLI")
    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser("search", help="Search for documents")
    search_parser.add_argument("index", type=str, help="Elasticsearch index name")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--from_", type=int, default=0, help="Offset for search results")

    get_doc_parser = subparsers.add_parser("get", help="Get a document by ID")
    get_doc_parser.add_argument("index", type=str, help="Elasticsearch index name")
    get_doc_parser.add_argument("doc_id", type=str, help="Document ID")

    reindex_parser = subparsers.add_parser("reindex", help="Reindex the documents")
    reindex_parser.add_argument("index", type=str, help="Elasticsearch index name")

    index_parser = subparsers.add_parser("index", help="Index a JSON file")
    index_parser.add_argument("index", type=str, help="Elasticsearch index name")
    index_parser.add_argument("json_file", type=str, help="Path to the JSON file")

    xml_parser = subparsers.add_parser("xml", help="Process XML search request")
    xml_parser.add_argument("xml_file", type=str, help="Path to the XML file")

    args = parser.parse_args()

    if args.command == "search":
        handle_search(args.index, args.query, args.from_)
    elif args.command == "get":
        get_document(args.index, args.doc_id)
    elif args.command == "reindex":
        reindex(args.index)
    elif args.command == "index":
        index_json_file(args.index, args.json_file)
    elif args.command == "xml":
        title, request = process_xml_request(args.xml_file)
        if title and request:
            print(f"Processed XML with title: {title} and request: {request}")
        else:
            print("Failed to process XML file.")
