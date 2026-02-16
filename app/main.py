from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from app.api.routes import router
from app.core.rate_limiter import limiter


app = FastAPI(
    title="Enterprise Document QA API",
    version="1.0.0",
)

# ---------------------------
# Rate limiting
# ---------------------------
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------
# Routes
# ---------------------------
app.include_router(router)


# ---------------------------
# Startup warmup
# ---------------------------
@app.on_event("startup")
def startup():
    print("✅ FastAPI startup complete — port is now open")

    # Auto-warm LLM + embeddings
    try:
        from app.core.model_manager import get_models
        print("🚀 Warming up models...")
        get_models()
        print("✅ Models warmed")
    except Exception as e:
        print(f"⚠️ Model warmup failed: {e}")


# ---------------------------
# Home page (Chat + Upload UI)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Ask My AI</title>

<style>
* { box-sizing: border-box; }

body {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
    max-width: 780px;
    margin: 40px auto;
    padding: 20px;
    color: #111827;
}

h1 { text-align: center; margin-bottom: 6px; }

.subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 24px;
}

.card {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

textarea {
    width: 100%;
    min-height: 90px;
    font-size: 16px;
    padding: 14px;
    border-radius: 10px;
    border: 1px solid #d1d5db;
    resize: vertical;
    outline: none;
}

textarea:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15);
}

input[type="file"],
input[type="text"] {
    width: 100%;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #d1d5db;
}

.actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 12px;
}

button {
    padding: 10px 22px;
    font-size: 16px;
    border-radius: 10px;
    border: none;
    background: #4f46e5;
    color: white;
    cursor: pointer;
}

button:disabled {
    background: #9ca3af;
    cursor: not-allowed;
}

.chat {
    margin-top: 24px;
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.bubble {
    max-width: 85%;
    padding: 14px 16px;
    border-radius: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
}

.user {
    align-self: flex-end;
    background: #4f46e5;
    color: white;
    border-bottom-right-radius: 4px;
}

.ai {
    align-self: flex-start;
    background: #f3f4f6;
    color: #111827;
    border-bottom-left-radius: 4px;
}

.cursor {
    animation: blink 1s infinite;
}

@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0; }
    100% { opacity: 1; }
}

.status {
    margin-top: 10px;
    font-size: 14px;
    color: #374151;
}
</style>
</head>

<body>

<h1>Ask My AI 🤖</h1>
<div class="subtitle">Upload documents and ask questions in real time</div>

<div class="card">
<h3>Upload Document (PDF or paste text)</h3>
<input type="file" id="docFile" accept=".pdf" /><br><br>
<textarea id="docText" placeholder="Or paste document text here..." style="height:120px;"></textarea><br>
<input id="source" placeholder="Optional file name" /><br><br>
<button id="uploadBtn" onclick="uploadDoc()">Upload Document</button>
<div id="uploadStatus" class="status"></div>
</div>

<div class="card">
<textarea id="question" placeholder="Ask a question about the document..."></textarea>
<div class="actions">
<button id="askBtn" onclick="ask()">Ask</button>
</div>
</div>

<div id="chat" class="chat"></div>

<script>
const USER_KEY = "user-key-456";

function uploadDoc() {
    const fileInput = document.getElementById("docFile");
    const textInput = document.getElementById("docText");
    const sourceInput = document.getElementById("source");
    const statusDiv = document.getElementById("uploadStatus");

    if(fileInput.files.length > 0){
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append("file", file);
        formData.append("source_file", sourceInput.value || file.name);

        fetch("/ingest", {
            method: "POST",
            headers: { "X-API-Key": USER_KEY },
            body: formData
        })
        .then(r=>r.json())
        .then(d=>statusDiv.textContent = d.status)
        .catch(()=>statusDiv.textContent = "Upload failed");
    }
    else if(textInput.value.trim() !== ""){
        fetch("/ingest", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-API-Key": USER_KEY
            },
            body: new URLSearchParams({
                text: textInput.value,
                source_file: sourceInput.value || "user_text"
            })
        })
        .then(r => r.json())
        .then(d => statusDiv.textContent = d.status)
        .catch(() => statusDiv.textContent = "Upload failed");
    }
    else{
        statusDiv.textContent = "Nothing to upload";
    }
}

function ask() {
    const questionInput = document.getElementById("question");
    const question = questionInput.value.trim();
    if (!question) return;

    const chat = document.getElementById("chat");
    const button = document.getElementById("askBtn");

    const userBubble = document.createElement("div");
    userBubble.className = "bubble user";
    userBubble.textContent = question;
    chat.appendChild(userBubble);

    const aiBubble = document.createElement("div");
    aiBubble.className = "bubble ai";
    aiBubble.innerHTML = '<span class="cursor">▍</span>';
    chat.appendChild(aiBubble);

    questionInput.value = "";
    button.disabled = true;

    fetch("/ask/stream", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-API-Key": USER_KEY
        },
        body: JSON.stringify({ question })
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        function read() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    aiBubble.innerHTML = aiBubble.textContent;
                    button.disabled = false;
                    return;
                }

                const chunk = decoder.decode(value);
                const lines = chunk.split("\\n\\n");

                lines.forEach(line => {
                    if (line.startsWith("data: ")) {
                        aiBubble.textContent += line.replace("data: ", "");
                        aiBubble.innerHTML = aiBubble.textContent + '<span class="cursor">▍</span>';
                    }
                });

                read();
            });
        }

        read();
    }).catch(() => {
        aiBubble.textContent = "❌ Error occurred.";
        button.disabled = false;
    });
}
</script>

</body>
</html>
"""
