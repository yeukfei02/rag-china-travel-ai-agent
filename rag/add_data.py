import json
import uuid
from vector_db.pinecone import pc, index_name


def add_data_to_pinecone():
    try:
        data_list = []

        json_files = ["data/china-travel-q&a.json"]

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)

                for item in data:
                    chunk_text = f"{item['question']} {item['answer']}" or ""

                    obj = {
                        "_id": str(uuid.uuid4()),
                        "chunk_text": chunk_text
                    }
                    data_list.append(obj)

        print(f"data_list = {data_list}")
        print(f"data_list length = {len(data_list)}")

        records = data_list

        # target the index
        target_index = pc.Index(index_name)

        # upsert the records into a namespace in batches of 96
        batch_size = 96
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            target_index.upsert_records("travel-planner-namespace", batch)

        # show stats
        stats = target_index.describe_index_stats()
        print(f"stats = {stats}")
    except Exception as e:
        print(f"add_data_to_pinecone error = {e}")


add_data_to_pinecone()
