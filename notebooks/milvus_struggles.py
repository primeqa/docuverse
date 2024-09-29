import pickle
import time
from docuverse.utils.timer import timer

from docuverse.utils import open_stream
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction
from pymilvus import (
    MilvusClient,
    DataType
)

cache_file = "/home/raduf/.local/share/elastic_ingestion/benchmark__beir_dev__quora____en__corpus.small.jsonl_512_100_True_all_gte-small.pickle.xz"
MODEL = "/home/raduf/sandbox2/docuverse/models/slate.30m.english.rtrvr-20240719T181101"
num_examples = 100

data = pickle.load(open_stream(cache_file))[:num_examples]

model = DenseEmbeddingFunction(MODEL)


embeddings = model.encode([d['text'] for d in data], show_progress_bar=True)
keys_to_keep = {"text"}

data_list = [{**{k: v for k, v in d.items() if k in keys_to_keep}, '_id': d['id'], 'qembedding': embeddings[i]} for i, d
             in enumerate(data)]
questions = [
    data_list[0]['text'],
    "How can I get free gems in Clash of Clans?",
    "How can I get free gems Clash of Clans?",
    "How do you feel when someone upvotes your answer on Quora?",
    "What are the best thriller movie in Hollywood?",
    "What should someone do to overcome anxiety?"
]

def test_search(vectors, vector_for_query=None, metric="IP", reingest=False, milvus_server_addr="test.db",
                use_connections=False, ingest_batch_size=-1):
    truncate_dim = 384
    collection_name = "test1"
    vector_field_name = "qembedding"
    if ingest_batch_size < 0:
        ingest_batch_size = len(vectors)

    if vector_for_query is None:
        entities = vectors
        test = vectors[0:3]
    else:
        entities = vectors
        if isinstance(vector_for_query, list) and isinstance(vector_for_query[0], dict):
            test = [d[vector_field_name] for d in vector_for_query]# [{vector_field_name: e} for e in vector_for_query]
        else:
            test = vector_for_query

    tm = timer()
    if use_connections:
        from pymilvus import connections
        client = connections
        init, host, port = milvus_server_addr.split(":")
        host = host.replace("//", "")
        client.connect(host=host, port=port)
        client1 = MilvusClient(milvus_server_addr)
    else:
        client = MilvusClient(milvus_server_addr)
        client1 = client


    if reingest or not client.has_collection(collection_name=collection_name):
        schema = client1.create_schema(auto_id=True, enable_dynamic_field=True, primary_field="id")

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=50000)
        schema.add_field(field_name="_id", datatype=DataType.VARCHAR, max_length=50000)
        schema.add_field(field_name=vector_field_name, datatype=DataType.FLOAT_VECTOR, dim=truncate_dim)

        index_params = client1.prepare_index_params()
        index_params.add_index(
            field_name=vector_field_name,
            index_type="FLAT",
            metric_type=metric,
            index_name=vector_field_name,
            params={"nlist": 1024}
        )

        client1.drop_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )
        for i in range(1, len(vectors), ingest_batch_size):
            client.insert(collection_name=collection_name, data=entities[i:i + ingest_batch_size])
        # insert_result = client.insert(collection_name=collection_name, data=entities)
        # print({k: v for k, v in insert_result.items() if k != 'ids'})
        client.load_collection(collection_name=collection_name)
        ingested_items = 0
        tm.mark()
        while ingested_items!=len(vectors):
            res = client.get_collection_stats(collection_name=collection_name)
            ingested_items = res["row_count"]
            print(f"Currently ingested items: {ingested_items}")
            time.sleep(10)
        print(f"Ingested in {tm.time_since_last_mark()} seconds.")
        print(client.list_indexes(collection_name=collection_name))
        print(client.describe_index(collection_name=collection_name, index_name=vector_field_name))

    return client.search(
        collection_name=collection_name,
        # data=[t[vector_field_name] for t in test],
        data=test,
        #data=test,
        search_params={"metric_type": metric, "params": {"nprobe": 100, "efSearch": 128}},
        # anns_field=vector_field_name,
        limit=10,
        output_fields=["text", "_id"],
    )

def print_answer(q, res):
    print(f"Question: {q['text'] if isinstance(q, dict) else q}")
    if len(res) == 0:
        print("  ** No results found. **")
    else:
        for r in res:
            print({'id': r['entity']['_id'], 'answer': r['entity']['text'], 'score': r['distance']})
        print("\n")

def test_setup(milvus_server_addr="test.db", reingest=False, use_connections=False):
    global questions, score
    print(f"Testing {milvus_server_addr}")
    questions = data_list
    answers = test_search(data_list, questions, reingest=reingest, milvus_server_addr=milvus_server_addr,
                          use_connections=use_connections)
    score = 0
    total = len(questions)

    for q, res in zip(questions, answers):
        if isinstance(q, dict):
            if q['_id'] in [r['entity']['_id'] for r in res[0:3]]:
                score += 1
            else:
                if len(res) > 0:
                    print_answer(q, res)
        else:
            print_answer(q, res)
    print(f"Score: {score * 1.0 / total:.2f}")

online_milvus = "http://localhost:19530"
file_milvus = "test.db"
use_connections = True
test_setup(online_milvus, reingest=False, use_connections=False)
# test_setup(file_milvus, reingest=False, use_connections=False)
