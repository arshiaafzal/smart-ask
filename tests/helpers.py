"""Small fakes shared by the unit tests."""

from types import SimpleNamespace

from smart_ask.domain import ExecutionRequest, ModelResult


def usage(prompt_tokens: int = 10, completion_tokens: int = 2):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def response(content: str, call_usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=call_usage or usage(),
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


class RecordingExecutor:
    captures_output = True

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def execute(self, request: ExecutionRequest) -> ModelResult:
        self.calls.append(request)
        return ModelResult(model=request.model, text=self.texts.pop(0))
