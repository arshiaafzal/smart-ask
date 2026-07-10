"""Authentication at the Claude protocol edge."""

from __future__ import annotations

from collections.abc import Mapping
import hmac


class AdapterAuthenticator:
    def __init__(self, token: str | None, *, required: bool):
        if required and (not isinstance(token, str) or not token.strip()):
            raise ValueError("adapter authentication token is required")
        self._token = token or ""
        self._required = required

    def authenticate(self, headers: Mapping[str, str]) -> bool:
        if not self._required:
            return True
        authorization = headers.get("authorization")
        bearer = None
        if authorization is not None:
            scheme, separator, value = authorization.partition(" ")
            if not separator or scheme.lower() != "bearer" or not value:
                return False
            bearer = value
        api_key = headers.get("x-api-key")
        if bearer is not None and api_key is not None and bearer != api_key:
            return False
        supplied = bearer if bearer is not None else api_key
        return supplied is not None and hmac.compare_digest(supplied, self._token)

