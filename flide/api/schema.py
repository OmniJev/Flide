"""Request and response types for the TypeSafe-compatible System One API.

Shapes follow TypeSafe's public OpenAPI description and the
behaviour of the official Python SDK (msgspec structs). Extra keys are tolerated everywhere;
Score legend and probability keys are integer-parseable strings.
"""
from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

Entry = Union[str, dict[str, Any], list[Any], None]   # a text field: a string or any JSON value

MAX_CHOICE_OPTIONS = 256
MAX_SCORE_LEVELS = 256


class _Lenient(BaseModel):
    """Keeps the keys it does not declare: a request's `images` or `video` reach the families that read them."""

    model_config = ConfigDict(extra="allow")


class NoulCriteria(_Lenient):
    """What true and what false mean for a noul question; either may be left out."""

    true: Entry = None
    false: Entry = None


class NoulQuestion(_Lenient):
    """A yes-or-no question. Its answer is the probability of true."""

    type: Literal["noul"]
    instructions: Entry = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Lenient):
    """One of several options. `criteria` maps each option's name to what it means (1 to 256 options); the answer
    gives every option's probability under the same names."""

    type: Literal["choice"]
    instructions: Entry = None
    criteria: dict[str, Entry]

    @model_validator(mode="after")
    def _check(self) -> "ChoiceQuestion":
        if not self.criteria:
            raise ValueError("choice question needs at least one option in criteria")
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice question has {len(self.criteria)} options, maximum is {MAX_CHOICE_OPTIONS}")
        return self


class ScoreQuestion(_Lenient):
    """A level on an ordered scale. `criteria` lists the levels from the lowest (1 to 256); the answer is keyed by
    each level's index as a string, "0" for the first."""

    type: Literal["score"]
    instructions: Entry = None
    criteria: list[Entry]

    @model_validator(mode="after")
    def _check(self) -> "ScoreQuestion":
        if not self.criteria:
            raise ValueError("score question needs at least one level in criteria")
        if len(self.criteria) > MAX_SCORE_LEVELS:
            raise ValueError(f"score question has {len(self.criteria)} levels, maximum is {MAX_SCORE_LEVELS}")
        return self


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


class SystemOneRequest(_Lenient):
    """One state and the questions asked about it. `state` is any JSON value (text, an object, chat messages), which
    the family renders; `questions` maps a name to each question, and the answers come back under the same names.
    `model` is the served model's name, one of its aliases or "jev-latest". Undeclared keys such as `images` are kept
    for the family that reads them."""

    state: Any = None
    model: str = "jev-latest"
    questions: dict[str, Question] = Field(discriminator=None)

    @model_validator(mode="after")
    def _check(self) -> "SystemOneRequest":
        if not self.questions:
            raise ValueError("at least one question is required")
        return self


class NoulAnswer(BaseModel):
    """`noul`: the probability of true, from 0 to 1."""

    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    """`choice`: the most probable option; `probabilities`: every option's, summing to 1; `confidence`: the top
    probability rescaled from uniform (0) to certain (1), as api/answers.py computes it."""

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    """`score`: the expected level index under `probabilities` (keyed "0", "1", ...); `legend`: the level each index
    stands for, as the question gave it; `confidence`: 1 when every bit of probability is on the most likely level,
    0 when it is spread as widely as the uniform distribution or wider."""

    type: Literal["score"] = "score"
    score: float
    legend: dict[str, Entry]
    probabilities: dict[str, float]
    confidence: float


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class Usage(BaseModel):
    """`input_tokens`: the tokens Flide read for the request; `output_tokens`: one per question answered."""

    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    """The answer to a SystemOneRequest: the served model's name, an answer under each question's name, and usage."""

    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(BaseModel):
    """One entry of GET /v1/models: the served model or one of its aliases."""

    name: str
    description: str
    release_date: str


class ModelMetadataList(BaseModel):
    """GET /v1/models: the served model first, then its aliases."""

    models: list[ModelMetadata]
