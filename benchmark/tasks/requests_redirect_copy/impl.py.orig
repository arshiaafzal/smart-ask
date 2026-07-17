"""
Minimal reproduction of requests redirect request-copy bug.
SWE-bench: psf__requests-1963

resolve_redirects reuses the *same* prepared request object across all
redirect hops, so mutations from one hop (e.g., method change after 303)
bleed into the next hop.  Each iteration must start from a fresh copy.
"""

from __future__ import annotations
from copy import copy
from dataclasses import dataclass, field


@dataclass
class PreparedRequest:
    method: str
    url: str
    headers: dict = field(default_factory=dict)
    body: bytes | None = None


@dataclass
class Response:
    status_code: int
    headers: dict = field(default_factory=dict)
    url: str = ""


def resolve_redirects(
    response: Response,
    request: PreparedRequest,
    redirects: list[Response],
) -> list[PreparedRequest]:
    """
    Follow a chain of redirect responses, yielding the PreparedRequest sent
    at each hop.

    Bug: `prepared_request` is never re-copied inside the loop, so method
    mutations from one hop carry over to subsequent hops.
    """
    history: list[PreparedRequest] = []
    prepared_request = request   # Bug: should copy(request) *inside* the loop

    for redirect in redirects:
        status = redirect.status_code

        # 303 See Other: switch to GET and drop body.
        if status == 303:
            prepared_request.method = "GET"
            prepared_request.body = None

        prepared_request.url = redirect.headers.get("Location", redirect.url)
        history.append(prepared_request)

    return history
