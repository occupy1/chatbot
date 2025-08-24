# Requirements (install once)
# pip install pymupdf4llm sentence-transformers chromadb ollama langchain
# Optional: pip install numpy

from __future__ import annotations
import os
from collections import deque

import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# ------------------------------
# 0) Config
# ------------------------------
FILE_PATH = "./test.pdf"                  # PDF 경로
#MODEL_NAME = "exaone3.5:latest"          # Ollama 모델명 (사전에 pull 필요)
MODEL_NAME = "llama3.1"
CHROMA_PATH = "./chroma_db"               # 벡터DB 저장 경로
COLLECTION_NAME = "rag_collection"
TOP_K = 2
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# e5 계열 임베딩 모델 (다국어)
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

# ------------------------------
# 1) Utils: PDF load & chunk
# ------------------------------
def load_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {os.path.abspath(file_path)}")
    print("PDF파일 로드..")
    pages = pymupdf4llm.to_markdown(file_path)
    # pymupdf4llm은 페이지별 텍스트 리스트를 반환
    return "".join(pages)


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)


# ------------------------------
# 2) Embed & ChromaDB index build
# ------------------------------
print("문서 로드 및 임베딩 시작...")
raw_text = load_pdf(FILE_PATH)
chunks = split_text(raw_text)

embedder = SentenceTransformer(EMBED_MODEL_NAME)
# e5 시리즈는 권장 포맷: 문서 임베딩 시 "passage: ", 질의 임베딩 시 "query: "
chunk_embeddings = embedder.encode([f"passage: {c}" for c in chunks], convert_to_tensor=False)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# 기존 인덱스가 있다면 중복 방지 차원에서 비워주고 다시 만듭니다(선택).
# 주석 해제하면 매번 새로 구축합니다.
# try:
#     existing = collection.get()
#     if existing and existing.get("ids"):
#         collection.delete(ids=existing["ids"])
# except Exception:
#     pass

ids = [f"chunk_{i+1}" for i in range(len(chunks))]
collection.add(
    ids=ids,
    embeddings=[emb.tolist() if hasattr(emb, "tolist") else emb for emb in chunk_embeddings],
    metadatas=[{"text": c} for c in chunks],
)


# ------------------------------
# 3) Message manager
# ------------------------------
class MessageManager:
    def __init__(self, maxlen: int = 10):
        self._system_msg = {"role": "system", "content": ""}
        self.queue: deque[dict] = deque(maxlen=maxlen)  # 최근 대화 10개

    def create_msg(self, role: str, content: str) -> dict:
        return {"role": role, "content": content}

    def set_system(self, content: str) -> None:
        self._system_msg = self.create_msg("system", content)

    def append_user(self, content: str) -> None:
        self.queue.append(self.create_msg("user", content))

    def append_assistant(self, content: str) -> None:
        self.queue.append(self.create_msg("assistant", content))

    def build_messages(self, retrieved_docs: list[str]) -> list[dict]:
        docs = "\n".join(retrieved_docs)
        return [
            self._system_msg,
            {
                "role": "system",
                "content": (
                    f"문서 내용: {docs}\n"
                    "질문에 대한 답변은 문서 내용을 기반으로 정확히 제공하시오."
                ),
            },
            *list(self.queue),
        ]


msg_manager = MessageManager()
msg_manager.set_system(
    (
        "가장 마지막 'user'의 'content'에 대해 답변한다."
        "질문에 답할 때는 'system' 메시지 중 '문서 내용'을 우선 참고한다."
        "개행은 문장 끝이나 항목 구분에만 사용한다."
    )
)


# ------------------------------
# 4) Retrieval & Generation
# ------------------------------
def retrieve_docs(query: str, top_k: int = TOP_K) -> list[str]:
    q_emb = embedder.encode(f"query: {query}", convert_to_tensor=False)
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    if not results.get("metadatas") or not results["metadatas"][0]:
        return ["관련 문서를 찾을 수 없습니다."]
    return [m["text"] for m in results["metadatas"][0]]


def generate_answer(query: str, retrieved_docs: list[str]) -> str:
    # 대화 이력에 사용자 질문 추가
    msg_manager.append_user(query)

    # Ollama로 스트리밍 생성
    messages = msg_manager.build_messages(retrieved_docs)
    print("답변: ", end="", flush=True)
    full = ""
    for event in ollama.chat(model=MODEL_NAME, messages=messages, stream=True):
        # 응답 형식: {"message": {"role": "assistant", "content": "..."}, "done": bool}
        chunk = event.get("message", {}).get("content", "")
        if chunk:
            print(chunk, end="", flush=True)
            full += chunk
    print()

    # 대화 이력에 모델 응답 추가
    if full:
        msg_manager.append_assistant(full)
    return full


# ------------------------------
# 5) Chat loop (CLI)
# ------------------------------
def chat_loop() -> None:
    print("RAG 챗봇 시작! 질문 입력 (종료하려면 'exit'):")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n챗봇 종료!")
            break
        if query.lower() == "exit":
            print("챗봇 종료!")
            break

        docs = retrieve_docs(query, top_k=TOP_K)
        _ = generate_answer(query, docs)


if __name__ == "__main__":
    # Ollama 준비 안내
    # 1) ollama serve
    # 2) ollama pull exaone3.5:latest  (또는 원하는 모델)
    chat_loop()
