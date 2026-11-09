"""Request and response types for the TypeSafe-compatible System One API.

Shapes follow TypeSafe's public OpenAPI description and the
behaviour of the official Python SDK (msgspec structs). Extra keys are tolerated everywhere;
Score legend and probability keys are integer-parseable strings.
"""
from __future__ import annotations

import math
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

Entry = Union[str, dict[str, Any], list[Any], None]
FiniteNumber = StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)]

MAX_CHOICE_OPTIONS = 256
MAX_SCORE_LEVELS = 256


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class NoulCriteria(_Lenient):
    true: Entry = None
    false: Entry = None


class NoulQuestion(_Lenient):
    type: Literal["noul"]
    instructions: Entry = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Lenient):
    type: Literal["choice"]
    instructions: Entry = None
    criteria: dict[str, Entry]
    multiple: bool = False

    @model_validator(mode="after")
    def _check(self) -> "ChoiceQuestion":
        if not self.criteria:
            raise ValueError("choice question needs at least one option in criteria")
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice question has {len(self.criteria)} options, maximum is {MAX_CHOICE_OPTIONS}")
        return self


class ScoreQuestion(_Lenient):
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


class ExtendedQuestion(_Lenient):
    type: Literal["text", "number", "date", "box", "boxes"]
    instructions: Entry = None
    range: tuple[FiniteNumber, FiniteNumber] | None = None
    step: FiniteNumber | None = Field(default=None, gt=0)
    unit: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ExtendedQuestion:
        if self.range is None:
            return self
        low, high = self.range
        try:
            width = high - low
            steps = width if self.step is None else width / self.step
            if all(math.isfinite(n) for n in (low, high, steps)):
                return self
        except OverflowError:
            pass
        raise ValueError("number range and step must produce a finite grid")


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion, ExtendedQuestion]


class SystemOneRequest(_Lenient):
    state: Any = None
    model: str = "jev-latest"
    questions: dict[str, Question] = Field(discriminator=None)

    @model_validator(mode="after")
    def _check(self) -> "SystemOneRequest":
        if not self.questions:
            raise ValueError("at least one question is required")
        return self


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, Entry]
    probabilities: dict[str, float]
    confidence: float


class MultipleChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choices: list[str]
    probabilities: dict[str, float]


class ExtractedAnswer(_Lenient):
    span: tuple[int, int] | None
    found: float


class TextAnswer(ExtractedAnswer):
    type: Literal["text"]
    text: str | None


class DateAnswer(ExtractedAnswer):
    type: Literal["date"]
    date: str | None
    text: str | None


class NumberAnswer(ExtractedAnswer):
    type: Literal["number"]
    number: FiniteNumber | None
    text: str | None
    unit: str | None


class EstimateAnswer(BaseModel):
    type: Literal["number"]
    number: float
    interval: list[float]
    unit: str | None


class BoxDetection(BaseModel):
    box: dict[str, float] | None
    found: float
    confidence: float


class BoxAnswer(BoxDetection):
    type: Literal["box"]


class BoxesAnswer(BaseModel):
    type: Literal["boxes"]
    boxes: list[BoxDetection]
    count: int
    at_least: bool
    presence: list[float]


Answer = Union[
    NoulAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    MultipleChoiceAnswer,
    TextAnswer,
    DateAnswer,
    NumberAnswer,
    EstimateAnswer,
    BoxAnswer,
    BoxesAnswer,
]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str


class ModelMetadataList(BaseModel):
    models: list[ModelMetadata]
