import json
import datetime
import numpy as np
from opensearchpy import helpers
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, RequestsHttpConnection

SERVER_URL = "http://luton:9200"
INDEX_NAME = "products_delete"

AMAZON_METADATA_PATH = "/home/nschwartz/Desktop/data_file.json"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def normalize_data(data):
    return data / np.linalg.norm(data, ord=2)


def load_file(file_path):
    try:
        json_objects = []
        with open(file_path, "r") as json_file:
            for line in json_file:
                data = json.loads(line)
                if type(data) is list:
                    for element in data:
                        if "imgUrl" in element and type(element["imgUrl"]) is not list:
                            imgUrl = list(json.loads(element["imgUrl"]).keys())[0]
                            element["imgUrl"] = imgUrl
                            json_objects.append(element)
                elif "imgUrl" in data and type(data["imgUrl"]) is not list:
                    imgUrl = list(json.loads(data["imgUrl"]).keys())[0]
                    data["imgUrl"] = imgUrl
                    json_objects.append(data)
        print("Done")
    finally:
        json_file.close()
    return json_objects


def get_client(server_url: str) -> OpenSearch:
    os_client_instance = OpenSearch('http://luton:9200', use_ssl=False, verify_certs=False,
                                    connection_class=RequestsHttpConnection)
    print("OS connected")
    print(datetime.datetime.now())
    return os_client_instance


def create_index(index_name: str, os_client: OpenSearch, metadata: np):
    mapping = {
        "mappings": {
            "properties": {
                "asin": {
                    "type": "keyword"
                },
                "description_vector": {
                    "type": "knn_vector",
                    "dimension": get_vector_dimension(metadata),
                },
                "item_image": {
                    "type": "keyword",
                },
                "text_field": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword_field": {
                            "type": "keyword"
                        }
                    }
                }
            }
        },
        "settings": {
            "index": {
                "number_of_shards": "1",
                "knn": "false",
                "number_of_replicas": "0"
            }
        }

    }
    os_client.indices.create(index=index_name, body=mapping)


def delete_index(index_name: str, os_client: OpenSearch):
    os_client.indices.delete(index_name)


def get_vector_dimension(metadata: list):
    title = metadata[0]["title"]
    embeddings = model.encode(title)
    return len(embeddings)


def store_index(index_name: str, data: np.array, metadata: list, os_client: OpenSearch):
    documents = []
    for index_num, vector in enumerate(data):
        metadata_line = metadata[index_num]
        text_field = metadata_line["title"]
        embedding = model.encode(text_field)
        norm_text_vector_np = normalize_data(embedding)
        document = {
            "_index": index_name,
            "_id": index_num,
            "asin": metadata_line["asin"],
            "description_vector": norm_text_vector_np.tolist(),
            "item_image": metadata_line["imgUrl"],
            "text_field": text_field
        }
        documents.append(document)
        if index_num % 1000 == 0 or index_num == len(data):
            helpers.bulk(os_client, documents, request_timeout=1800)
            documents = []
            print(f"bulk {index_num} indexed successfully")
            os_client.indices.refresh(INDEX_NAME)

    os_client.indices.refresh(INDEX_NAME)


os_client = get_client(SERVER_URL)
metadata = load_file(AMAZON_METADATA_PATH)
create_index(INDEX_NAME, os_client, metadata)
store_index(INDEX_NAME, metadata, metadata, os_client)
