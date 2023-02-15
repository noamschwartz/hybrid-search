import numpy as np
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, RequestsHttpConnection

SERVER_URL = "http://luton:9200"
# INDEX_NAME = "products"
INDEX_NAME = "shopee_products"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def normalize_bm25_formula(score, max_score):
    return score / max_score


def normalize_bm25(bm_results):
    hits = (bm_results["hits"]["hits"])
    max_score = bm_results["hits"]["max_score"]
    for hit in hits:
        hit["_score"] = normalize_bm25_formula(hit["_score"], max_score)
    bm_results["hits"]["max_score"] = hits[0]["_score"]
    bm_results["hits"]["hits"] = hits
    return bm_results


def run_queries(os_client):
    while True:
        query = input("Enter your vector search query: ")
        vector_boost_level = float(input("Enter how much vector search boost you want to apply: "))
        if query == "exit":
            break
        else:
            bm25_boost_level = float(input("Enter how much keyword search boost you want to apply: "))
            apu_request_body = {
                "size": 20,
                "query": {
                    "gsi_knn": {
                        "field": "description_vector",
                        "vector": get_vector_sentence_transformers(query).tolist(),
                    }
                },
                "_source": ["asin", "text_field", "item_image"],
            }
            # reduce the scores by 1 when using cpu
            cpu_request_body = {
                "size": 20,
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "knn_score",
                            "lang": "knn",
                            "params": {
                                "field": "description_vector",
                                "query_value": get_vector_sentence_transformers(query).tolist(),
                                "space_type": "cosinesimil"
                            }
                        }
                    }
                },
                "_source": ["asin", "text_field", "item_image"],
            }

            bm25_query = {
                "size": 20,
                "query": {
                    "match": {
                        "text_field": query
                    }
                },
                "_source": ["asin", "text_field", "item_image"],
            }
            vector_search_results = os_client.search(body=apu_request_body, index=INDEX_NAME)
            bm25_results = os_client.search(body=bm25_query, index=INDEX_NAME)
            bm25_results = normalize_bm25(bm25_results)
            combined_results = interpolate_results(vector_search_results["hits"]["hits"],
                                                   bm25_results["hits"]["hits"])
            sorted_elements = apply_boost(combined_results, vector_boost_level, bm25_boost_level)

            result_data_dictionary = extract_results_data(vector_search_results["hits"]["hits"],
                                                          bm25_results["hits"]["hits"])
            construct_response(result_data_dictionary, sorted_elements)


def extract_results_data(vector_data, bm25_data):
    result_data_dictionary = {}
    for vector_hit in vector_data:
        product_id = vector_hit["_source"]["asin"]
        img_url = vector_hit["_source"]["item_image"]
        text_description = vector_hit["_source"]["text_field"]
        result_data_dictionary[product_id] = [img_url, text_description]
    for bm25_hit in bm25_data:
        product_id = bm25_hit["_source"]["asin"]
        img_url = bm25_hit["_source"]["item_image"]
        text_description = bm25_hit["_source"]["text_field"]
        result_data_dictionary[product_id] = [img_url, text_description]
    return result_data_dictionary


def construct_response(result_data_dictionary, sorted_elements):
    for index, sorted_element in enumerate(sorted_elements):
        print(index + 1, result_data_dictionary[sorted_element])


def get_vector_sentence_transformers(text_input):
    return model.encode(text_input)


def normalize_data(data):
    return data / np.linalg.norm(data, ord=2)


def get_client(server_url: str) -> OpenSearch:
    os_instance = OpenSearch('http://luton:9200', use_ssl=False, verify_certs=False,
                             connection_class=RequestsHttpConnection)
    # print("OS connected")
    return os_instance


def get_min_score(common_elements, elements_dictionary):
    if len(common_elements):
        return min([min(v) for v in elements_dictionary.values()])
    else:
        # No common results - assign arbitrary minimum score value
        return 0.01


def interpolate_results(vector_hits, bm25_hits):
    # gather all product ids
    bm25_ids_list = []
    vector_ids_list = []
    for hit in bm25_hits:
        bm25_ids_list.append(hit["_source"]["asin"])
    for hit in vector_hits:
        vector_ids_list.append(hit["_source"]["asin"])
    # find common product ids
    common_results = set(bm25_ids_list) & set(vector_ids_list)
    results_dictionary = dict((key, []) for key in common_results)
    for common_result in common_results:
        for index, vector_hit in enumerate(vector_hits):
            if vector_hit["_source"]["asin"] == common_result:
                results_dictionary[common_result].append(vector_hit["_score"])
        for index, BM_hit in enumerate(bm25_hits):
            if BM_hit["_source"]["asin"] == common_result:
                results_dictionary[common_result].append(BM_hit["_score"])
    min_value = get_min_score(common_results, results_dictionary)
    # assign minimum value scores for all unique results
    for vector_hit in vector_hits:
        if vector_hit["_source"]["asin"] not in common_results:
            new_scored_element_id = vector_hit["_source"]["asin"]
            results_dictionary[new_scored_element_id] = [min_value]
    for BM_hit in bm25_hits:
        if BM_hit["_source"]["asin"] not in common_results:
            new_scored_element_id = BM_hit["_source"]["asin"]
            results_dictionary[new_scored_element_id] = [min_value]

    return results_dictionary


def apply_boost(combined_results, vector_boost_level, bm25_boost_level):
    for element in combined_results:
        if len(combined_results[element]) == 1:
            combined_results[element] = combined_results[element][0] * vector_boost_level + \
                                        combined_results[element][0] * bm25_boost_level
        else:
            combined_results[element] = combined_results[element][0] * vector_boost_level + \
                                        combined_results[element][1] * bm25_boost_level
    # sort the results based on the new scores
    sorted_results = [k for k, v in sorted(combined_results.items(), key=lambda item: item[1], reverse=True)]
    return sorted_results


def main():
    os_client = get_client(SERVER_URL)
    run_queries(os_client)


if __name__ == "__main__":
    main()
