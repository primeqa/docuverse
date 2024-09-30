import pickle
import time
from docuverse.utils.timer import timer
from tqdm import trange
from docuverse.utils import open_stream
from docuverse.utils.embeddings.dense_embedding_function import DenseEmbeddingFunction
from pymilvus import (
    MilvusClient,
    Collection,
    DataType,
    connections,
    utility, FieldSchema, CollectionSchema
)

cache_file = "/home/raduf/.local/share/elastic_ingestion/benchmark__beir_dev__quora____en__corpus.small.jsonl_512_100_True_all_gte-small.pickle.xz"
MODEL = "/home/raduf/sandbox2/docuverse/models/slate.30m.english.rtrvr"
num_examples = -1

data = pickle.load(open_stream(cache_file))[:num_examples]

model = DenseEmbeddingFunction(MODEL)

conns = {
    "quora_docuverse": ("beir_quora_small_milvus_dense_512_100_08292024", "embeddings", "COSINE", 10),
    "test": ("test3", "qembedding", "IP", 10)
}
test_instance_keys = ['collection_name', 'vector_field_name', 'metric', 'ingest_batch_size']

test_instance = conns["test"]


embeddings = model.encode([d['text'] for d in data], show_progress_bar=True)
keys_to_keep = {"text"}


def refactor_data(embeddings):
    return [
        {
            **{k: v for k, v in d.items() if k in keys_to_keep},
            '_id': d['id'],
            test_instance[1]: embeddings[i]
        } for i, d in enumerate(data)
    ]


data_list = refactor_data(embeddings)
questions = [
    data_list[0]['text'],
    "How can I get free gems in Clash of Clans?",
    "How can I get free gems Clash of Clans?",
    "How do you feel when someone upvotes your answer on Quora?",
    "What are the best thriller movie in Hollywood?",
    "What should someone do to overcome anxiety?"
]

# def test_search_old(vectors, vector_for_query=None, metric="IP", reingest=False, milvus_server_addr="test.db",
#                 use_connections=False, ingest_batch_size=-1, vector_field_name="qembedding", collection_name="test3"):
#     truncate_dim = 384
#
#     if ingest_batch_size < 0:
#         ingest_batch_size = len(vectors)
#
#     if vector_for_query is None:
#         entities = vectors
#         test = vectors[0:3]
#     else:
#         entities = vectors
#         if isinstance(vector_for_query, list) and isinstance(vector_for_query[0], dict):
#             test = [d[vector_field_name] for d in vector_for_query]# [{vector_field_name: e} for e in vector_for_query]
#         else:
#             test = vector_for_query
#
#     if use_connections:
#         client = connections
#         init, host, port = milvus_server_addr.split(":")
#         host = host.replace("//", "")
#         client.connect(host=host, port=port)
#         client1 = MilvusClient(milvus_server_addr)
#     else:
#         client = MilvusClient(milvus_server_addr)
#         client1 = client
#
#
#     if reingest or not client.has_collection(collection_name=collection_name):
#         print("Ingesting documents.")
#         schema = client1.create_schema(auto_id=True, enable_dynamic_field=True, primary_field="id")
#
#         schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
#         schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=50000)
#         schema.add_field(field_name="_id", datatype=DataType.VARCHAR, max_length=50000)
#         schema.add_field(field_name=vector_field_name, datatype=DataType.FLOAT_VECTOR, dim=truncate_dim)
#
#         index_params = client1.prepare_index_params()
#         index_params.add_index(
#             field_name=vector_field_name,
#             index_type="FLAT",
#             metric_type=metric,
#             index_name=vector_field_name,
#             params={"nlist": 1024}
#         )
#
#         client1.drop_collection(collection_name=collection_name)
#         client.create_collection(
#             collection_name=collection_name, schema=schema, index_params=index_params
#         )
#         print("Ingesting ..", end=" ")
#         for i in range(0, len(vectors), ingest_batch_size):
#             client.insert(collection_name=collection_name, data=entities[i:i + ingest_batch_size])
#         client.flush()
#         print("done.")
#         # insert_result = client.insert(collection_name=collection_name, data=entities)
#         # print({k: v for k, v in insert_result.items() if k != 'ids'})
#         client.load_collection(collection_name=collection_name)
#         if milvus_server_addr.find("localhost")>=0:
#             ingested_items = 0
#             client = MilvusClient(milvus_server_addr)
#             connections.connect(host="localhost", port=19530)
#             utility.wait_for_index_building_complete(collection_name=collection_name, index_name=vector_field_name)
#             tm = timer()
#             start = time.time()
#             while ingested_items < len(vectors)-1:
#                 res = client.get_collection_stats(collection_name=collection_name)
#                 ingested_items = res["row_count"]
#                 print(f"{tm.time_since_beginning()}: Currently ingested items: {ingested_items}")
#                 time.sleep(10)
#             print(f"Ingested in {tm.time_since_beginning()} seconds.")
#         print(f"Index list for {collection_name}: {client.list_indexes(collection_name=collection_name)}")
#         print(f"Index stats: {client.describe_index(collection_name=collection_name, index_name=vector_field_name)}")
#
#     return client.search(
#         collection_name=collection_name,
#         # data=[t[vector_field_name] for t in test],
#         data=test,
#         #data=test,
#         search_params={"metric_type": metric, "params": {"nprobe": 100, "efSearch": 128}},
#         # anns_field=vector_field_name,
#         limit=10,
#         output_fields=["text", "_id"],
#     )

def test_search(vectors, vector_for_query=None, metric="IP", reingest=False, milvus_server_addr="test.db",
                use_connections=False, ingest_batch_size=-1, vector_field_name="qembedding", collection_name="test3"):

    def has_collection(client, use_connections, collection_name):
        if use_connections:
            return utility.has_collection(collection_name)
        else:
            return client.has_collection(collection_name)
    truncate_dim = 384

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

    if use_connections:
        init, host, port = milvus_server_addr.split(":")
        host = host.replace("//", "")
        connections.connect(host=host, port=port)
        client = MilvusClient(milvus_server_addr)
    else:
        client = MilvusClient(milvus_server_addr)

    if reingest or not has_collection(collection_name=collection_name, use_connections=use_connections, client=client):
        print("Ingesting documents.")
        index_dict = {
            "field_name":vector_field_name,
            "index_type":"FLAT",
            "metric_type":metric,
            "index_name":vector_field_name,
            "params":{"nlist": 1024}
        }
        if use_connections:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="_id", dtype=DataType.VARCHAR, is_primary=False, max_length=1000, auto_id=False),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name=vector_field_name, dtype=DataType.FLOAT_VECTOR, dim=truncate_dim)
            ]
            schema = CollectionSchema(fields=fields, description="crap")
            utility.drop_collection(collection_name)

            collection = Collection(name=collection_name, schema=schema)
            print("Ingesting ..", end=" ")
            for i in trange(0, len(vectors), ingest_batch_size):
                collection.insert(data=entities[i:i + ingest_batch_size])
            collection.create_index(field_name=vector_field_name, index_params=index_dict)
            print("done.")
            collection.flush()
            utility.wait_for_index_building_complete(collection_name=collection_name, index_name=vector_field_name)
            print("Done waiting.")
            print(f"Index list for {collection_name}: {utility.list_indexes(collection_name=collection_name)}")
            collection.load()
        else:
            schema = client.create_schema(auto_id=True, enable_dynamic_field=True, primary_field="id")

            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=50000)
            schema.add_field(field_name="_id", datatype=DataType.VARCHAR, max_length=50000)
            schema.add_field(field_name=vector_field_name, datatype=DataType.FLOAT_VECTOR, dim=truncate_dim)

            index_params = client.prepare_index_params()
            index_params.add_index(
                **index_dict
                # field_name=vector_field_name,
                # index_type="FLAT",
                # metric_type=metric,
                # index_name=vector_field_name,
                # params={"nlist": 1024}
            )

            client.drop_collection(collection_name=collection_name)

            client.create_collection(
                collection_name=collection_name, schema=schema, index_params=index_params, consistency_level="Bounded"
            )
            print("Ingesting ..", end=" ")

            for i in trange(0, len(vectors), ingest_batch_size):
                client.insert(collection_name=collection_name, data=entities[i:i + ingest_batch_size])

            print("done.")
        # insert_result = client.insert(collection_name=collection_name, data=entities)
        # print({k: v for k, v in insert_result.items() if k != 'ids'})
            client.load_collection(collection_name=collection_name)
            if milvus_server_addr.find("localhost")>=0:
                ingested_items = 0
                client = MilvusClient(milvus_server_addr)
                connections.connect(host="localhost", port=19530)
                utility.wait_for_index_building_complete(collection_name=collection_name, index_name=vector_field_name)
                tm = timer()
                start = time.time()
                while False and ingested_items < len(vectors)-1:
                    res = client.get_collection_stats(collection_name=collection_name)
                    ingested_items = res["row_count"]
                    print(f"{tm.time_since_beginning()}: Currently ingested items: {ingested_items}")
                    time.sleep(10)
                print(f"Ingested in {tm.time_since_beginning()} seconds.")
            print(f"Index list for {collection_name}: {client.list_indexes(collection_name=collection_name)}")
            print(f"Index stats: {client.describe_index(collection_name=collection_name, index_name=vector_field_name)}")

    return do_search(client, collection_name, metric, test, use_connections, vector_field_name)


def do_search(client, collection_name, metric, test, use_connections, vector_field_name):
    search_params = {"metric_type": metric, "params": {"nprobe": 100, "efSearch": 128}}
    if use_connections:
        collection = Collection(name=collection_name)
        return collection.search(
            data=test,
            param=search_params,
            anns_field=vector_field_name,
            limit=10,
            output_fields=["text", "_id"],
        )
    else:
        return client.search(
            collection_name=collection_name,
            data=test,
            search_params=search_params,
            limit=10,
            output_fields=["text", "_id"]
        )


def print_answer(q, res):
    print(f"Question: {q['text'] if isinstance(q, dict) else q}")
    if len(res) == 0:
        print("  ** No results found. **")
    else:
        for r in res:
            print({'id': r['entity']['_id'], 'answer': r['entity']['text'], 'score': r['distance']})
        print("\n")

def test_setup(milvus_server_addr="test.db", questions=None, **kwargs):
    print(f"Testing {milvus_server_addr}")

    questions = data_list if questions is None else questions
    answers = test_search(data_list, questions, milvus_server_addr=milvus_server_addr, **kwargs)
    score = 0
    total = len(questions)

    score = 0
    for q, res in zip(questions, answers):
        if isinstance(q, dict):
            if q['_id'] in [r.fields['_id'] if hasattr(r, 'fields') else r['entity']['_id'] for r in res[0:3]]:
                score += 1
            else:
                if len(res) > 0:
                    print_answer(q, res)
        else:
            print_answer(q, res)
    print(f"Score: {score * 1.0 / total:.2f}")

online_milvus = "http://localhost:19530"
file_milvus = "test.db"
use_connections = False
# keys_to_keep = {"text"}
# data_list = [{**{k: v for k, v in d.items() if k in keys_to_keep}, '_id': d['id'], test_instance[1]: embeddings[i]} for i, d in
#              enumerate(data)]

test_setup(online_milvus, data_list, reingest=True, use_connections=use_connections,
           **{k: v for k, v in zip(test_instance_keys, test_instance)}
           )
# test_setup(online_milvus, reingest=False, use_connections=False)
# test_setup(file_milvus, reingest=True, use_connections=False)
