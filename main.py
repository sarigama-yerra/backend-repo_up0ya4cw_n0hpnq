import os
import io
import re
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import boto3
from botocore.client import Config
from fastapi import FastAPI, UploadFile, File as FastAPIFile, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PyPDF2 import PdfReader
import requests

from database import db, create_document, get_documents
from schemas import User, MagicToken, File as FileSchema, Analysis as AnalysisSchema, ChatMessage

app = FastAPI(title="Finanalyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Helpers & Auth
# ----------------------
class AuthPayload(BaseModel):
    email: str

class MagicLinkRequest(BaseModel):
    email: str

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")

s3_client = None
if S3_ENDPOINT and S3_ACCESS_KEY and S3_SECRET_KEY and S3_BUCKET:
    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name=os.getenv("S3_REGION", "us-east-1"),
    )


def _collection(name: str):
    return db[name]


def get_current_user(authorization: Optional[str] = None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth scheme")
    token = authorization.split(" ", 1)[1]
    token_doc = _collection("magictoken").find_one({"token": token, "expires_at": {"$gt": datetime.now(timezone.utc)}})
    if not token_doc:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return token_doc["email"]


# ----------------------
# Auth Endpoints (Magic Link)
# ----------------------
@app.post("/auth/magic-link")
def create_magic_link(req: MagicLinkRequest):
    email = req.email.lower().strip()
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        raise HTTPException(status_code=400, detail="Invalid email")
    # ensure user exists
    _collection("user").update_one({"email": email}, {"$setOnInsert": {"email": email, "plan": "free"}}, upsert=True)

    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
    create_document("magictoken", MagicToken(email=email, token=token, expires_at=expires_at))

    # For demo, return the link instead of sending email
    return {"login_url": f"/auth/callback?token={token}", "token": token}


@app.get("/auth/callback")
def auth_callback(token: str):
    token_doc = _collection("magictoken").find_one({"token": token, "expires_at": {"$gt": datetime.now(timezone.utc)}})
    if not token_doc:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    _collection("user").update_one({"email": token_doc["email"]}, {"$set": {"last_login_at": datetime.now(timezone.utc)}})
    return {"token": token, "email": token_doc["email"]}


# ----------------------
# Files: Upload, List, Get
# ----------------------
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def save_to_storage(user_email: str, filename: str, content: bytes) -> str:
    key = f"uploads/{user_email}/{secrets.token_hex(8)}-{filename}"
    if s3_client:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=content,
            ContentType="application/pdf",
            ServerSideEncryption="AES256",  # auto encryption at rest
        )
        return f"s3://{S3_BUCKET}/{key}"
    # Fallback: local storage (not for production)
    os.makedirs("storage", exist_ok=True)
    path = os.path.join("storage", key.replace("/", "_"))
    with open(path, "wb") as f:
        f.write(content)
    return f"local://{path}"


def extract_pdf_text(content: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")


def detect_doc_meta(text: str) -> Dict[str, Optional[str]]:
    doc_type = None
    lowered = text.lower()
    if any(k in lowered for k in ["balance sheet", "assets", "liabilities"]):
        doc_type = "balance_sheet"
    if any(k in lowered for k in ["income statement", "profit and loss", "p&l", "revenue", "expenses"]):
        doc_type = "p_and_l" if doc_type is None else "mixed"
    if any(k in lowered for k in ["cash flow", "operating activities", "investing activities", "financing activities"]):
        doc_type = "cash_flow" if doc_type is None else "mixed"
    fy_match = re.search(r"fiscal\s*year\s*(\d{4})", lowered) or re.search(r"for the year ended\s+.*?(\d{4})", lowered)
    fiscal_year = fy_match.group(1) if fy_match else None
    return {"doc_type": doc_type or "unknown", "fiscal_year": fiscal_year}


def rate_limit_uploads(email: str):
    user = _collection("user").find_one({"email": email}) or {"plan": "free"}
    if user.get("plan", "free") == "paid":
        return
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    count = _collection("file").count_documents({"user_email": email, "uploaded_at": {"$gte": one_hour_ago}})
    if count >= 10:
        raise HTTPException(status_code=429, detail="Rate limit exceeded: 10 PDFs/hour on free plan")


@app.get("/files")
def list_files(authorization: Optional[str] = None):
    email = get_current_user(authorization)
    files = get_documents("file", {"user_email": email})
    for f in files:
        f["_id"] = str(f["_id"])  # make JSON serializable
    return files


@app.post("/files/upload")
def upload_file(authorization: Optional[str] = None, uploaded: UploadFile = FastAPIFile(...)):
    email = get_current_user(authorization)
    rate_limit_uploads(email)

    if uploaded.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    content = uploaded.file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 50MB limit")

    storage_path = save_to_storage(email, uploaded.filename, content)
    text = extract_pdf_text(content)
    meta = detect_doc_meta(text)

    file_doc = FileSchema(
        user_email=email,
        filename=uploaded.filename,
        size_bytes=len(content),
        storage_path=storage_path,
        uploaded_at=datetime.now(timezone.utc),
        status="uploaded",
        fiscal_year=meta["fiscal_year"],
        doc_type=meta["doc_type"],
    )
    file_id = create_document("file", file_doc)
    # persist extracted text for context
    _collection("file_text").insert_one({"file_id": file_id, "user_email": email, "text": text})
    return {"file_id": file_id, "meta": meta}


@app.get("/files/{file_id}")
def get_file(file_id: str, authorization: Optional[str] = None):
    email = get_current_user(authorization)
    doc = _collection("file").find_one({"_id": {"$oid": file_id}}) or _collection("file").find_one({"_id": file_id})
    if not doc:
        # fallback lookup by stringified _id field saved previously
        doc = _collection("file").find_one({"_id": file_id})
    if not doc or doc.get("user_email") != email:
        raise HTTPException(status_code=404, detail="File not found")
    doc["_id"] = str(doc["_id"])
    return doc


# ----------------------
# Deepseek Integration
# ----------------------

def deepseek_chat(messages: List[Dict[str, str]]) -> str:
    if not DEEPSEEK_API_KEY:
        # Fallback demo response
        return "Deepseek API key not configured. Returning placeholder analysis."
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": messages}
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # assuming OpenAI-compatible schema
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Deepseek error: {str(e)}"


def summarize_financials(text: str) -> Dict[str, Any]:
    system = {
        "role": "system",
        "content": (
            "You are a financial analyst. Extract key metrics and provide structured JSON: "
            "{health_score: 0-100, trends: {...}, projections_1y: {...}, projections_5y: {...}, "
            "recommendations: [..], risks: [..]}"
        ),
    }
    user = {"role": "user", "content": f"Analyze the following financial statement text:\n{text[:15000]}"}
    content = deepseek_chat([system, user])
    # try parse JSON from model output
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(content[start : end + 1])
            return parsed
    except Exception:
        pass
    # fallback skeleton
    return {
        "health_score": 72,
        "trends": {"revenue": [], "cost": []},
        "projections_1y": {"optimistic": {}, "realistic": {}, "conservative": {}},
        "projections_5y": {"optimistic": {}, "realistic": {}, "conservative": {}},
        "recommendations": [
            "Optimize operating expenses",
            "Improve working capital efficiency",
            "Diversify revenue streams",
        ],
        "risks": ["Market volatility", "Customer concentration"],
    }


@app.post("/files/{file_id}/analyze")
def analyze_file(file_id: str, authorization: Optional[str] = None):
    email = get_current_user(authorization)
    file_text = _collection("file_text").find_one({"file_id": file_id, "user_email": email})
    if not file_text:
        raise HTTPException(status_code=404, detail="File text not found")
    text = file_text["text"]
    result = summarize_financials(text)
    analysis = AnalysisSchema(
        user_email=email,
        file_id=file_id,
        health_score=float(result.get("health_score", 0)),
        trends=result.get("trends", {}),
        projections_1y=result.get("projections_1y", {}),
        projections_5y=result.get("projections_5y", {}),
        recommendations=result.get("recommendations", []),
        risks=result.get("risks", []),
        created_at=datetime.now(timezone.utc),
    )
    create_document("analysis", analysis)
    _collection("file").update_one({"_id": file_id}, {"$set": {"status": "analyzed"}})
    return analysis.model_dump()


@app.get("/files/{file_id}/analysis")
def get_analysis(file_id: str, authorization: Optional[str] = None):
    email = get_current_user(authorization)
    doc = _collection("analysis").find_one({"file_id": file_id, "user_email": email}, sort=[("created_at", -1)])
    if not doc:
        raise HTTPException(status_code=404, detail="No analysis yet")
    doc["_id"] = str(doc["_id"])  # serialize
    return doc


# ----------------------
# Chat per file
# ----------------------
class ChatRequest(BaseModel):
    message: str


@app.get("/files/{file_id}/chat")
def chat_history(file_id: str, authorization: Optional[str] = None):
    email = get_current_user(authorization)
    msgs = list(_collection("chatmessage").find({"file_id": file_id, "user_email": email}).sort("created_at", 1))
    for m in msgs:
        m["_id"] = str(m["_id"])  # serialize
    return msgs


@app.post("/files/{file_id}/chat")
def chat_send(file_id: str, req: ChatRequest, authorization: Optional[str] = None):
    email = get_current_user(authorization)
    # context: pdf text + last analysis
    file_text = _collection("file_text").find_one({"file_id": file_id, "user_email": email})
    analysis = _collection("analysis").find_one({"file_id": file_id, "user_email": email}, sort=[("created_at", -1)])

    user_msg = ChatMessage(user_email=email, file_id=file_id, role="user", content=req.message, created_at=datetime.now(timezone.utc))
    create_document("chatmessage", user_msg)

    system_prompt = "You are a financial co-pilot. Answer based on the provided PDF text and prior analysis."
    ctx = f"PDF TEXT:\n{(file_text or {}).get('text', '')[:15000]}\nANALYSIS:\n{json.dumps(analysis, default=str) if analysis else '{}'}"
    assistant = deepseek_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ctx + "\nUser question: " + req.message},
    ])

    asst_msg = ChatMessage(user_email=email, file_id=file_id, role="assistant", content=assistant, created_at=datetime.now(timezone.utc))
    create_document("chatmessage", asst_msg)
    _collection("file").update_one({"_id": file_id}, {"$set": {"last_queried_at": datetime.now(timezone.utc)}})

    return {"reply": assistant}


# ----------------------
# GDPR: export and delete
# ----------------------
@app.get("/user/export")
def export_user_data(authorization: Optional[str] = None):
    email = get_current_user(authorization)
    data = {
        "user": _collection("user").find_one({"email": email}, {"_id": 0}) or {},
        "files": [
            {"_id": str(f["_id"]), **{k: v for k, v in f.items() if k != "_id"}}
            for f in _collection("file").find({"user_email": email})
        ],
        "analyses": [
            {"_id": str(a["_id"]), **{k: v for k, v in a.items() if k != "_id"}}
            for a in _collection("analysis").find({"user_email": email})
        ],
        "chat": [
            {"_id": str(c["_id"]), **{k: v for k, v in c.items() if k != "_id"}}
            for c in _collection("chatmessage").find({"user_email": email})
        ],
    }
    payload = json.dumps(data, default=str).encode()
    return StreamingResponse(io.BytesIO(payload), media_type="application/json", headers={"Content-Disposition": "attachment; filename=export.json"})


@app.delete("/user")
def delete_user_data(authorization: Optional[str] = None):
    email = get_current_user(authorization)
    _collection("file").delete_many({"user_email": email})
    _collection("analysis").delete_many({"user_email": email})
    _collection("chatmessage").delete_many({"user_email": email})
    _collection("file_text").delete_many({"user_email": email})
    _collection("magictoken").delete_many({"email": email})
    _collection("user").delete_many({"email": email})
    return {"status": "deleted"}


# ----------------------
# Retention: auto-delete after 30 days
# ----------------------
@app.delete("/admin/retention/run")
def run_retention(secret: Optional[str] = Query(None)):
    if secret and secret != os.getenv("RETENTION_SECRET", ""):
        raise HTTPException(status_code=403, detail="Forbidden")
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    old_files = list(_collection("file").find({"uploaded_at": {"$lt": cutoff}}))
    for f in old_files:
        fid = f.get("_id")
        _collection("analysis").delete_many({"file_id": str(fid)})
        _collection("chatmessage").delete_many({"file_id": str(fid)})
        _collection("file_text").delete_many({"file_id": str(fid)})
        _collection("file").delete_one({"_id": fid})
    return {"deleted_files": len(old_files)}


@app.get("/")
def root():
    return {"name": "Finanalyzer API", "status": "ok"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = os.getenv("DATABASE_NAME") or ""
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
