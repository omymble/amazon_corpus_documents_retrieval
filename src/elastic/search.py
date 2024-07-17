import json
from pprint import pprint
import os

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


from elasticsearch import Elasticsearch


class Search:
    def __init__(self):
        self.es = Elasticsearch()

    def index_document(self, index, doc_id, document):
        self.es.index(index=index, id=doc_id, document=document)

    def search(self, index, query, size=5, from_=0):
        return self.es.search(index=index, body={"query": query, "from": from_, "size": size})

    def reindex(self, index):
        if self.es.indices.exists(index=index):
            self.es.indices.delete(index=index)
        self.es.indices.create(index=index)

    def retrieve_document(self, index, doc_id):
        return self.es.get(index=index, id=doc_id)


# class Search:
#     def __init__(self):
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")
#         self.es = Elasticsearch(
#             'http://localhost:9200',
#             # cloud_id=os.environ["ELASTIC_CLOUD_ID"],
#             # api_key=os.environ["ELASTIC_API_KEY"],
#
#         )
#         client_info = self.es.info()
#         print("Connected to Elasticsearch!")
#         pprint(client_info.body)
#
#     def create_index(self):
#         self.es.indices.delete(index="my_documents", ignore_unavailable=True)
#         self.es.indices.create(
#             index="my_documents",
#             mappings={
#                 "properties": {
#                     "embedding": {
#                         "type": "dense_vector",
#                     }
#                 }
#             },
#         )
#
#     def get_embedding(self, text):
#         return self.model.encode(text)
#
#     def insert_document(self, document):
#         return self.es.index(
#             index="my_documents",
#             document={
#                 **document,
#                 "embedding": self.get_embedding(document["summary"]),
#             },
#         )
#
#     def insert_documents(self, documents):
#         operations = []
#         for document in documents:
#             operations.append({"index": {"_index": "my_documents"}})
#             operations.append(
#                 {
#                     **document,
#                     "embedding": self.get_embedding(document["summary"]),
#                 }
#             )
#         return self.es.bulk(operations=operations)
#
#     def reindex(self):
#         self.create_index()
#         with open("data.json", "rt") as f:
#             documents = json.loads(f.read())
#         return self.insert_documents(documents)
#
#     def search(self, **query_args):
#         return self.es.search(index="my_documents", **query_args)
#
#     def retrieve_document(self, id):
#         return self.es.get(index="my_documents", id=id)