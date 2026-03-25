"""API key authentication middleware."""

import os

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer()

API_KEY = os.environ.get("API_KEY", "")


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """Validate Bearer token against configured API_KEY."""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured on server")
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials
