import csv
import datetime
import time
import chromadb
import traceback

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    # Load metadata from CSV
    csv_file_path = "COA_OpenData.csv"
    
    metadata_list = []
    documents = []
    ids=[]
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        id = 0
        for row in reader:
            doc_text=row["HostWords"]
            documents.append(doc_text)
            
            # 嘗試解析不同格式的日期字串
            try:
                dt = datetime.datetime.strptime(row["CreateDate"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                dt = datetime.datetime.strptime(row["CreateDate"], "%Y-%m-%d")
            # 將 datetime 對象轉換為時間戳格式（秒）
            timestamp = int(dt.timestamp())

            metadata = {
                "file_name": "COA_OpenData.csv",
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": timestamp
            }
            metadata_list.append(metadata)
            ids.append(str(id))
            id += 1

    # Insert metadata into ChromaDB
    collection.add(
        documents=documents,
        metadatas=metadata_list,
        ids=ids
    )
    
    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

if __name__ == "__main__": 
    last_chunk = generate_hw01()
    print(last_chunk) 