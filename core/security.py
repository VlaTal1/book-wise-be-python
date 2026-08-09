import logging
from functools import lru_cache

import jwt
from jwt import PyJWKClient

from core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_jwk_client() -> PyJWKClient:
    jwks_url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
    logger.info(f"Loading JWKS from: {jwks_url}")
    return PyJWKClient(jwks_url)


def verify_token(token: str) -> str:
    """Валідує Supabase JWT (ES256, той самий JWKS, що й у Java) і повертає
    userId (subject). Кидає jwt.PyJWTError при недійсному токені.

    audience="authenticated" — стандартне значення aud у токенах Supabase для
    залогінених користувачів. PyJWT (на відміну від Nimbus, який використовує
    Java) вимагає явно вказати очікувану audience, якщо вона є в токені —
    інакше кидає InvalidAudienceError."""
    signing_key = _get_jwk_client().get_signing_key_from_jwt(token)
    claims = jwt.decode(token, signing_key.key, algorithms=["ES256"], audience="authenticated")
    return claims["sub"]
