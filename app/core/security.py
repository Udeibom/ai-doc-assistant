from fastapi import HTTPException, Request

#In production, load from env or secrets manager
API_KEYS = {
    "admin-key-123": "admin",
    "user-key-456": "user",
}

def get_role_from_request(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key"
        )

    role = API_KEYS.get(api_key)

    if not role:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return role

def require_user_or_admin(role: str):
    if role not in ("user", "admin"):
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions"
        )

def require_admin(role: str):
    if role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
