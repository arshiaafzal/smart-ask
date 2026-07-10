"""Small fakes shared by the unit tests."""

from types import SimpleNamespace

from smart_ask.domain import ExecutionRequest, ModelResult


def usage(prompt_tokens: int = 10, completion_tokens: int = 2):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def response(content: str, call_usage=None, *, model: str | None = None):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=call_usage or usage(),
    )


def responses_response(
    content: str,
    call_usage=None,
    *,
    model: str | None = None,
    status: str = "completed",
):
    return SimpleNamespace(
        model=model,
        status=status,
        output_text=content,
        output=[],
        usage=call_usage or SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            input_tokens_details=SimpleNamespace(
                cached_tokens=0,
                cache_write_tokens=0,
            ),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
        incomplete_details=None,
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        next_response = self.responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)
        self.responses = FakeCompletions(responses)


class RecordingExecutor:
    captures_output = True

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def execute(self, request: ExecutionRequest) -> ModelResult:
        self.calls.append(request)
        return ModelResult(model=request.model, text=self.texts.pop(0))
