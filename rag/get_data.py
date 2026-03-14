from rag.vector_db.pinecone import pc, index_name


def get_pinecone_data(query):
    data_list = []

    try:
        # target the index
        target_index = pc.Index(index_name)

        results = target_index.search(
            namespace="travel-planner-namespace",
            query={
                "top_k": 10,
                "inputs": {
                    'text': query
                }
            }
        )

        # print the results
        for hit in results['result']['hits']:
            print(f"hit = {hit}")
            print(
                f"Id: {hit['_id']}, Score: {hit['_score']}, Text: {hit['fields']['chunk_text']}")

            data = {
                "id": hit['_id'],
                "score": hit['_score'],
                "text": hit['fields']['chunk_text']
            }
            data_list.append(data)
    except Exception as e:
        print(f"get_pinecone_data error = {e}")

    return data_list
