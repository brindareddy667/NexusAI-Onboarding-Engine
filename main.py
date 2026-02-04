import os
import sqlite3
import PyPDF2
import docx
import chromadb
import io
import json
import re
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import contextmanager
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

# --- 1. INITIALIZATION & CONFIG ---
# Load environment variables from .env file
load_dotenv()

app = FastAPI()

DB_PATH = "onboarding.db"
UPLOAD_FOLDER = "demo_docs"
SUBMISSION_FOLDER = "employee_submissions"

# Retrieve API Key securely from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ensure directories exist
for folder in [UPLOAD_FOLDER, SUBMISSION_FOLDER, "static", "templates"]:
    os.makedirs(folder, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# AI Setup: Local Embeddings for Privacy + Groq for Speed
local_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="nexus_vfinal", embedding_function=local_ef)
groq_client = Groq(api_key=GROQ_API_KEY)

# --- 2. DATABASE LAYER ---
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try: yield conn
    finally: conn.close()

def init_db():
    with get_db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users(email TEXT PRIMARY KEY, name TEXT, role TEXT, score INTEGER DEFAULT 0)")
        conn.execute("""CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_email TEXT, title TEXT, 
            day TEXT, description TEXT, status TEXT, type TEXT, 
            ref_file TEXT, submission_link TEXT)""")
        conn.commit()
init_db()

def extract_text(content, filename):
    text = ""
    try:
        if filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            for p in reader.pages: text += p.extract_text() or ""
        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs])
    except Exception as e: print(f"Extraction Error: {e}")
    return text

# --- 3. PAGE ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request): return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard(request: Request): return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/employee", response_class=HTMLResponse)
async def serve_employee(request: Request): return templates.TemplateResponse("employee.html", {"request": request})

@app.get("/roadmap", response_class=HTMLResponse)
async def serve_roadmap(request: Request): return templates.TemplateResponse("roadmap.html", {"request": request})

@app.get("/assessment", response_class=HTMLResponse)
async def serve_assessment(request: Request): return templates.TemplateResponse("assessment.html", {"request": request})

# --- 4. CORE API LOGIC ---

@app.post("/upload")
async def upload_docs(files: list[UploadFile] = File(...)):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for f in files:
        content = await f.read()
        text = extract_text(content, f.filename)
        with open(os.path.join(UPLOAD_FOLDER, f.filename), "wb") as b: b.write(content)
        if text.strip():
            chunks = splitter.split_text(text)
            collection.add(documents=chunks, metadatas=[{"source": f.filename}]*len(chunks), ids=[f"{f.filename}_{i}" for i in range(len(chunks))])
    return {"status": "success"}

@app.post("/generate-roadmap")
async def gen_roadmap(name: str = Form(...), email: str = Form(...), role: str = Form(...), resume_file: UploadFile = File(...)):
    res_text = extract_text(await resume_file.read(), resume_file.filename)
    r_res = collection.query(query_texts=[f"Procedures for {role}"], n_results=3)
    g_res = collection.query(query_texts=["Company Policies"], n_results=2)
    files = list(set([m['source'] for m in r_res['metadatas'][0]] + [m['source'] for m in g_res['metadatas'][0]]))
    
    prompt = f"Role: {role}. Resume: {res_text}. Available Files: {files}. Create a 7-day JSON plan. Tasks must use ref_file from the list. Format: [{{'day': 1, 'title': '...', 'desc': '...', 'type': 'READ_DOC', 'ref_file': '...'}}]"
    comp = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
    tasks = json.loads(re.search(r'\[.*\]', comp.choices[0].message.content, re.DOTALL).group())
    
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO users (email, name, role) VALUES (?,?,?)", (email, name, role))
        conn.execute("DELETE FROM tasks WHERE user_email=?", (email,))
        for t in tasks:
            conn.execute("INSERT INTO tasks (user_email, title, day, description, status, type, ref_file, submission_link) VALUES (?,?,?,?,?,?,?,?)",
                          (email, t['title'], str(t['day']), t['desc'], "PENDING", t.get('type', 'READ_DOC'), t.get('ref_file', ''), ""))
        conn.commit()
    return {"status": "success"}

@app.get("/document-content")
async def get_doc_content(filename: str):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path): return {"content": "Not found."}
    with open(path, "rb") as f: raw = extract_text(f.read(), filename)
    # Force AI to generate a specific challenge block
    prompt = f"Transform this SOP into a structured lesson. End with a clearly labeled 'TECHNICAL CHALLENGE' section for the hire to complete: {raw}"
    comp = groq_client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
    return {"content": comp.choices[0].message.content}

@app.post("/tasks/submit")
async def submit_task(task_id: int = Form(...), link: str = Form(None), file: UploadFile = File(None)):
    submission_data = link if link else "Manual Submission"
    if file:
        file_path = os.path.join(SUBMISSION_FOLDER, f"task_{task_id}_{file.filename}")
        with open(file_path, "wb") as buffer: buffer.write(await file.read())
        submission_data = f"Uploaded: {file.filename}"

    with get_db() as conn:
        conn.execute("UPDATE tasks SET status='DONE', submission_link=? WHERE id=?", (submission_data, task_id))
        conn.commit()
    return {"status": "success"}

@app.get("/assessment-status")
async def check_status(email: str):
    with get_db() as conn:
        tasks = conn.execute("SELECT status FROM tasks WHERE user_email=?", (email,)).fetchall()
    unlocked = all(t[0] == "DONE" for t in tasks) if tasks else False
    return {"unlocked": unlocked}

@app.get("/quiz")
async def get_quiz(email: str):
    with get_db() as conn: 
        row = conn.execute("SELECT role FROM users WHERE email=?", (email,)).fetchone()
        role = row[0] if row else "Employee"
    res = collection.query(query_texts=[f"Requirements for {role}"], n_results=5)
    prompt = f"""
    Context: {''.join(res['documents'][0])}. 
    Generate 15 questions (10 MCQ, 5 Scenario) for {role}. 
    Return JSON: {{ "questions": [ {{ "type":"mcq", "question":"...", "options":[], "answer":"..." }}, {{ "type":"scenario", "question":"..." }} ] }}
    """
    comp = groq_client.chat.completions.create(messages=[{"role":"user","content":prompt}], model="llama-3.3-70b-versatile", response_format={"type": "json_object"})
    return JSONResponse(content=json.loads(comp.choices[0].message.content))

@app.get("/tasks")
async def get_tasks(email: str):
    with get_db() as conn:
        cursor = conn.execute("SELECT id, title, day, description, status, type, ref_file, submission_link FROM tasks WHERE user_email=?", (email,))
        return [{"id": r[0], "title": r[1], "day": r[2], "desc": r[3], "status": r[4], "type": r[5], "ref_file": r[6], "link": r[7]} for r in cursor.fetchall()]

@app.get("/hr-data")
async def hr_data():
    out = []
    with get_db() as conn:
        users = conn.execute("SELECT name, email, role, score FROM users").fetchall()
        for u in users:
            tasks = conn.execute("SELECT status FROM tasks WHERE user_email=?", (u[1],)).fetchall()
            done = len([t for t in tasks if t[0] == 'DONE'])
            prog = int((done / len(tasks)) * 100) if tasks else 0
            status = "PASSED" if (prog == 100 and u[3] >= 9) else "PENDING"
            out.append({"name": u[0], "email": u[1], "role": u[2], "progress": prog, "score": f"{u[3]}/15", "status": status})
    return out

@app.post("/save-result")
async def save_res(email: str = Form(...), score: int = Form(...)):
    with get_db() as conn:
        conn.execute("UPDATE users SET score=? WHERE email=?", (score, email))
        conn.commit()
    return {"ok": True}

@app.post("/chat")
async def chat_bot(request: Request):
    data = await request.json()
    res = collection.query(query_texts=[data['question']], n_results=3)
    comp = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": f"Context: {''.join(res['documents'][0])}"}, {"role": "user", "content": data['question']}],
        model="llama-3.3-70b-versatile"
    )
    return {"answer": comp.choices[0].message.content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)