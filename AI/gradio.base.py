import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
# 載入 Chroma 向量資料庫
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    chunk_size=1000,
)
vectorstore = Chroma(persist_directory="c:/Users/88698/Desktop/AI/---/AI/chroma", embedding_function=embeddings, collection_name="baseball-glossary")

def ai_glossary_bot(message, history):
    # 查詢最相關的術語
    results = vectorstore.similarity_search(message, k=1)
    if results:
        return results[0].page_content
    else:
        return "找不到相關術語，請換個問題試試。"

demo = gr.ChatInterface(fn=ai_glossary_bot, type="messages", examples=["什麼是滿壘？", "請解釋三振", "打帶跑是什麼意思？"], title="棒球術語 AI 助理")
demo.launch()