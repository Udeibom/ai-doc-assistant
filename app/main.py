from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from app.api.routes import router
from app.core.rate_limiter import limiter
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Enterprise Document QA API",
    version="1.0.0"
)

#Rate limiting
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(router)

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
                background: #f7f7f7;
                max-width: 700px;
                margin: 50px auto;
            }
            h1 {
                text-align: center;
            }
            textarea {
                width: 100%;
                height: 100px;
                font-size: 16px;
                padding: 10px;
            }
            button {
                margin-top: 10px;
                padding: 10px 20px;
                font-size: 16px;
            }
            #answer {
                margin-top: 20px;
                white-space: pre-wrap;
                background: #fff;
                padding: 15px;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Ask My AI 📄🤖</h1>
        <textarea id="question" placeholder="Ask a question..."></textarea>
        <br>
        <button onclick="ask()">Ask</button>

        <div id="answer"></div>

        <script>
            async function ask() {
                const question = document.getElementById("question").value;
                const answerDiv = document.getElementById("answer");
                answerDiv.textContent = "Thinking...";

                const response = await fetch("/ask", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-API-Key": "user-key-456"
                    },
                    body: JSON.stringify({ question })
                });

                const data = await response.json();
                answerDiv.textContent = data.answer;
            }
        </script>
    </body>
    </html>
    """
