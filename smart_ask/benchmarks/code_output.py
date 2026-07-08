"""Code-output normalization shared by executable benchmark suites."""


def extract_code(text: str) -> str:
    """Return an outer fenced code block, or the unfenced text unchanged.

    Model transports intentionally preserve provider output. Code benchmarks
    own the narrower policy of extracting the first outer Markdown code block.
    Leading indentation inside the block is retained.
    """

    stripped = text.strip()
    if not stripped.startswith("```"):
        return text.rstrip()

    lines = stripped.splitlines()
    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "```"
        ),
        len(lines),
    )
    return "\n".join(lines[1:closing_index]).rstrip()
