"""The Gemini models every script uses, in one place, with an automatic fallback.

PRIMARY  = gemini-flash-latest       (env GEMINI_MODEL overrides)
FALLBACK = gemini-flash-lite-latest  (env GEMINI_FALLBACK_MODEL overrides)

The gemini-2.5-* text models these scripts used are no longer available to new keys. When the primary
model is unavailable (retired, overloaded, out of quota, a gateway error), a call is retried once on the
fallback. Any other error (bad request, bad key, safety) is raised as it is.

    import gemini_models as gm
    r = gm.generate(client, model=gm.PRIMARY, contents=..., config=...)   # client: a google.genai.Client
    for model in gm.chain(gm.PRIMARY): ...                                # REST callers: try each in turn
"""
import os
import sys

# The -latest aliases exist only in API v1beta; google-genai 3.x calls v1 unless told otherwise. Clients read this
# when they are created, so import this module before making a genai.Client.
os.environ.setdefault("GOOGLE_GENAI_API_VERSION", "v1beta")

PRIMARY = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
FALLBACK = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest")

# what "unavailable" looks like: HTTP codes and google.genai / REST status names
_UNAVAILABLE_CODES = {404, 429, 500, 502, 503, 504}
_UNAVAILABLE_WORDS = ("NOT_FOUND", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL",
                      "no longer available", "high demand", "overloaded", "upstream request failed")


def chain(model=None):
    """[model, FALLBACK] without duplicates: the order to try."""
    model = model or PRIMARY
    return [model] + ([FALLBACK] if FALLBACK and FALLBACK != model else [])


def unavailable(e):
    """True when an error means 'this model can't answer now' (switch models), not 'this request is wrong'."""
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if isinstance(code, int) and code in _UNAVAILABLE_CODES:
        return True
    s = str(e)
    return any(w in s for w in _UNAVAILABLE_WORDS) or any(f"HTTP Error {c}" in s for c in _UNAVAILABLE_CODES)


def generate(client, model=None, **kw):
    """client.models.generate_content(model=..., **kw), falling back to FALLBACK when the model is unavailable."""
    err = None
    for m in chain(model):
        try:
            return client.models.generate_content(model=m, **kw)
        except Exception as e:  # noqa: BLE001
            if not unavailable(e):
                raise
            err = e
            if m != chain(model)[-1]:
                print(f"(gemini: {m} unavailable: {str(e)[:100]}; falling back to {FALLBACK})", file=sys.stderr)
    raise err
