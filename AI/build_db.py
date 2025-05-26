import os
import json
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
# ...existing code...

load_dotenv()  # 載入 .env 中的 API KEY

# 1. 載入 JSON
with open("glossary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 準備文件格式
docs = []
for item in data:
    content = f"【術語】{item['term']}（{item['english']}）\n定義：{item['definition']}\n範例：{item['example']}"
    metadata = {"term": item["term"], "english": item["english"]}
    docs.append(Document(page_content=content, metadata=metadata))

# 3. 轉換成向量
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    chunk_size=1000,  # 新增這一行
)
vectorstore = Chroma.from_documents(docs, embedding=embeddings, collection_name="baseball-glossary")

# 4. 儲存資料庫
vectorstore.persist()
print("✅ Chroma 向量資料庫建立完成")
