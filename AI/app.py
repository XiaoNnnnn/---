import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
import gradio as gr

# 載入 .env 環境變數
load_dotenv()

# 初始化向量資料庫
embedding = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    chunk_size=1000,
)
db = Chroma(
    embedding_function=embedding,
    persist_directory="./chroma",  # 預設資料夾，可依你實際存放位置
    collection_name="baseball-glossary"
)

# 初始化語言模型（回應生成）
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

# 建立 RAG QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),  # 向量資料庫作為檢索器
    return_source_documents=True  # 如果你想顯示來源
)

# Gradio 介面函式
def rag_chat(message, history):
    response = qa_chain.invoke({"query": message})
    return response["result"]


# 啟動介面
demo = gr.ChatInterface(
    fn=rag_chat,
    title="九局通",
    examples=["什麼是安打？", "英文中 'double play' 是什麼意思？", "請解釋三振的定義"],
    type="messages"
)

demo.launch()
