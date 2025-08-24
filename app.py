# app.py
# -------------------------------------------------------------
# 1) pip install fastapi uvicorn pymupdf4llm sentence-transformers chromadb ollama langchain
# 2) ollama serve && ollama pull llama3.1 # (또는 원하는 모델)
# 3) uvicorn app:app --reload
# -------------------------------------------------------------
from __future__ import annotations
import os, json
from typing import List, Literal, Optional
from collections import deque
from urllib.parse import unquote
import time

import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

from fastapi import FastAPI, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

SSE_READY = {}


# ------------------------------
# Config
# ------------------------------
FILE_PATH = "./test.pdf"
MODEL_NAME = "llama3.1:latest" # ex) llama3.1, mistral, gemma 등 (ollama pull 필요)
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
TOP_K = 2
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
HEARTBEAT_SEC = 15


# ------------------------------
# PDF → Chunk
# ------------------------------
def load_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {os.path.abspath(file_path)}")
    pages = pymupdf4llm.to_markdown(file_path)
    return "".join(pages)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
raw_text = load_pdf(FILE_PATH)
chunks = splitter.split_text(raw_text)


# ------------------------------
# Embedding & Vector DB
# ------------------------------
embedder = SentenceTransformer(EMBED_MODEL_NAME)
chunk_embeddings = embedder.encode([f"passage: {c}" for c in chunks], convert_to_tensor=False)


client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


ids = [f"chunk_{i+1}" for i in range(len(chunks))]
# 중복 add를 피하려면 최초 한 번만 구축하거나, 필요시 기존 ID 삭제 로직 추가
try:
    existing = collection.get()
    existing_ids = set(existing.get("ids", []))
except Exception:
    existing_ids = set()


new_ids, new_embs, new_metas = [], [], []
for i, (cid, c, emb) in enumerate(zip(ids, chunks, chunk_embeddings)):
    if cid not in existing_ids:
        new_ids.append(cid)
        new_embs.append(emb.tolist() if hasattr(emb, "tolist") else emb)
        new_metas.append({"text": c})
if new_ids:
    collection.add(ids=new_ids, embeddings=new_embs, metadatas=new_metas)


# ------------------------------
# Message Manager
# ------------------------------
class MessageManager:
    def __init__(self, maxlen: int = 10):
        self._system_msg = {"role": "system", "content": ""}
        self.queue: deque[dict] = deque(maxlen=maxlen)


    def set_system(self, content: str):
        self._system_msg = {"role": "system", "content": content}


    def build_messages(self, history: List[dict], retrieved_docs: List[str]):
        docs = "".join(retrieved_docs)
        return [
            self._system_msg,
            {
                "role": "system",
                "content": (
                    f"문서 내용: {docs}"
                    "질문에 대한 답변은 문서 내용을 기반으로 정확히 제공하시오."
                ),
            },
            *history,
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
# Retrieval
# ------------------------------
def retrieve_docs(query: str, top_k: int = TOP_K) -> List[str]:
    q_emb = embedder.encode(f"query: {query}", convert_to_tensor=False)
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    if not results.get("metadatas") or not results["metadatas"][0]:
        return ["관련 문서를 찾을 수 없습니다."]
    return [m["text"] for m in results["metadatas"][0]]

# ------------------------------
# FastAPI App
# ------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 개발 편의. 운영에서는 도메인 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: Literal["user","assistant","system"]
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = []
    model: Optional[str] = MODEL_NAME

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# public 폴더를 정적 디렉토리로 마운트
app.mount("/static", StaticFiles(directory="public"), name="static")

# 루트에서 index.html 서빙
@app.get("/", response_class=HTMLResponse)
def root():
    with open("public/index.html", "r", encoding="utf-8") as f:
        return f.read()

# (선택) 파비콘
@app.get("/favicon.ico")
def favicon():
    return StaticFiles(directory="public").lookup_path("favicon.ico")[0]

from fastapi.responses import StreamingResponse, PlainTextResponse

@app.post("/api/chat")
async def api_chat(body: ChatRequest = Body(...)):
    query = body.query.strip()
    history = [m.dict() for m in (body.history or [])]
    model = body.model or MODEL_NAME

    retrieved = retrieve_docs(query, top_k=TOP_K)
    messages = msg_manager.build_messages(history + [{"role":"user","content":query}], retrieved)

    def gen():
        for event in ollama.chat(model=model, messages=messages, stream=True):
            chunk = event.get("message", {}).get("content", "")
            if chunk:
                yield (chunk)
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

# ★ 신규: SSE(EventSource) 엔드포인트 (GET)
# - query, model, history(JSON 문자열)를 쿼리스트링으로 받습니다.
@app.get("/api/chat_sse")
async def api_chat_sse(
    query: str = Query(...),
    model: str = Query(MODEL_NAME),
    history: str = Query("[]"), # URL-encoded JSON string
    sid: str = Query(...),       # ✅ 추가: 세션 ID
):
    # history 파싱
    try:
        hist_list = json.loads(unquote(history))
        if not isinstance(hist_list, list):
            hist_list = []
    except Exception:
        hist_list = []

    retrieved = retrieve_docs(query, top_k=TOP_K)
    messages = msg_manager.build_messages(hist_list + [{"role": "user", "content": query}], retrieved)
    
    # (1) Pre-flight: 모델/서버 점검
    try:
        _ = ollama.chat(model=model, messages=messages[-1:], stream=False)  # 아주 짧게 점검
    except Exception as e:
        # 모델 미설치/ollama 미가동 등은 SSE 대신 즉시 오류 반환 (디버깅에 명확)
        return PlainTextResponse(f"[precheck failed] {type(e).__name__}: {e}", status_code=503)

    def sse_gen():
        # 시작 신호 (원하면 프론트에서 typing on 트리거에 사용)
        yield "retry: 10000\n"
        yield "event: start\ndata: start\n\n"
        
        # 핸드셰이크 대기 (최대 10초)
        t0 = time.time()
        while not SSE_READY.get(sid, False):     # 존재 확인만
            if time.time() - t0 > 10:
                yield "event: model_error\ndata: handshake timeout\n\n"
                yield "event: done\ndata: done\n\n"
                return
            yield ": waiting\n\n"
            time.sleep(0.05)
        # 루프 탈출 시에만 제거
        SSE_READY.pop(sid, None)
        time.sleep(0.1)  # 100ms 정도 (환경에 따라 50~200ms)
        
        last_ping = time.time()

        try:
            for event in ollama.chat(model=model, messages=messages, stream=True):
                now = time.time()
                if now - last_ping > HEARTBEAT_SEC:
                    yield ": ping\n\n" # 주석 라인은 SSE 하트비트로 사용됨
                    last_ping = now
                
                chunk = event.get("message", {}).get("content", "")
                if not chunk:
                    continue
                
                # 디버그 로그 (서버 콘솔에서 바로 확인)
                print("[chunk]", repr(chunk[:80]))

                #for line in chunk.replace("\r\n","\n").replace("\r","\n").split("\n"):
                #    yield f"data: {line}\n"
                #yield "\n"      # ← 이벤트 종료(빈 줄)
                
                chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
                yield f"data: {chunk}\n\n"   # ★ 원청크 그대로
        except Exception as e:
            print("[SSE error]", type(e).__name__, str(e))
            yield f"event: model_error\ndata: {type(e).__name__}: {str(e)}\n\n"
        finally:
            yield "event: done\ndata: done\n\n"
        
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no", # nginx 등에서 버퍼링 방지
    }
    return StreamingResponse(sse_gen(), media_type="text/event-stream", headers=headers)
    
# (진단용) SSE 핑 테스트 엔드포인트
@app.get("/api/ping_sse")
async def ping_sse():
    def gen():
        yield "event: start / data: start"
        for i in range(5):
            yield f"data: ping {i}"
            time.sleep(1)
        yield "event: done / data: done"
        
    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

@app.post("/api/sse_ready")
async def sse_ready(payload: dict):
    sid = payload.get("sid")
    if not sid:
        return {"ok": False, "error": "sid required"}
    SSE_READY[sid] = True
    return {"ok": True}