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
# Startup check
# ---------------------------
@app.on_event("startup")
def startup():
    print("✅ FastAPI startup complete — port is now open")


# ---------------------------
# Home page (Streaming Chat UI)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Ask My AI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            max-width: 700px;
            margin: 40px auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 90px;
            font-size: 16px;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
        button {
            margin-top: 10px;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 6px;
            border: none;
            background: #4f46e5;
            color: white;
            cursor: pointer;
        }
        button:disabled {
            background: #999;
        }
        #answer {
            margin-top: 20px;
            white-space: pre-wrap;
            background: white;
            padding: 16px;
            border-radius: 6px;
            min-height: 120px;
        }
        .confidence {
            margin-top: 10px;
            font-size: 14px;
            color: #555;
        }
    </style>
</head>
<body>
    <h1>Ask My AI 📄🤖</h1>

    <textarea id="question" placeholder="Ask a question about the document..."></textarea>
    <br>
    <button id="askBtn" onclick="ask()">Ask</button>

    <div id="answer"></div>
    <div id="confidence" class="confidence"></div>

<script>
function ask() {
    const question = document.getElementById("question").value.trim();
    if (!question) return;

    const answerDiv = document.getElementById("answer");
    const confidenceDiv = document.getElementById("confidence");
    const button = document.getElementById("askBtn");

    answerDiv.textContent = "";
    confidenceDiv.textContent = "";
    button.disabled = true;

    fetch("/ask/stream", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-API-Key": "user-key-456"
        },
        body: JSON.stringify({ question })
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        function read() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    button.disabled = false;
                    return;
                }

                const chunk = decoder.decode(value);
                const lines = chunk.split("\\n\\n");

                lines.forEach(line => {
                    if (line.startsWith("data: ")) {
                        answerDiv.textContent += line.replace("data: ", "");
                    }
                    if (line.startsWith("event: metadata")) {
                        confidenceDiv.textContent = line.replace("event: metadata\\ndata: ", "");
                    }
                });

                read();
            });
        }

        read();
    }).catch(() => {
        answerDiv.textContent = "❌ Error occurred.";
        button.disabled = false;
    });
}
</script>
</body>
</html>
"""


# ---------------------------
# Admin page (Document Upload)
# ---------------------------
@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Admin – Upload Document</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            max-width: 700px;
            margin: 40px auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
        }
        textarea, input {
            width: 100%;
            font-size: 16px;
            padding: 10px;
            margin-top: 10px;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
        textarea {
            height: 180px;
        }
        button {
            margin-top: 15px;
            padding: 10px 24px;
            font-size: 16px;
            border-radius: 6px;
            border: none;
            background: #16a34a;
            color: white;
            cursor: pointer;
        }
        #status {
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>

<h1>Admin Document Upload 📄</h1>

<input id="source" placeholder="Source file name (e.g. contract_v2.pdf)" />
<textarea id="text" placeholder="Paste document text here..."></textarea>

<button onclick="upload()">Upload Document</button>

<div id="status"></div>

<script>
async function upload() {
    const text = document.getElementById("text").value.trim();
    const source_file = document.getElementById("source").value.trim();
    const status = document.getElementById("status");

    if (!text || !source_file) {
        status.textContent = "❌ Please fill in all fields.";
        return;
    }

    status.textContent = "Uploading...";

    try {
        const response = await fetch("/admin/ingest", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-API-Key": "admin-key-123"
            },
            body: JSON.stringify({ text, source_file })
        });

        const data = await response.json();

        if (!response.ok) {
            status.textContent = "❌ " + (data.detail || "Upload failed");
        } else {
            status.textContent = "✅ Document uploaded successfully!";
            document.getElementById("text").value = "";
            document.getElementById("source").value = "";
        }
    } catch (err) {
        status.textContent = "❌ Network error.";
    }
}
</script>

</body>
</html>
"""
