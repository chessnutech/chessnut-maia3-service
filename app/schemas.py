from __future__ import annotations

# SPDX-License-Identifier: AGPL-3.0-only
#
# AGPL-3.0 compliance boundary: these API schemas are part of the independently
# deployable Maia 3 inference microservice.

from pydantic import BaseModel, Field


class BotMoveRequest(BaseModel):
    fen: str
    moves: list[str] = Field(default_factory=list)
    model: str = "maia3-5m"
    elo: int = 1500
    multipv: int = 5
    temperature: float = 1.0
    top_p: float = 1.0


class MoveCandidate(BaseModel):
    move: str
    san: str = ""
    probability: float = 0.0
    wdl: list[int] = Field(default_factory=list)
    centipawns: int = 0


class BotMoveResponse(BaseModel):
    move: str
    san: str = ""
    probability: float = 0.0
    model: str = "maia3-5m"
    elo: int = 1500
    candidates: list[MoveCandidate] = Field(default_factory=list)


class ReviewGameRequest(BaseModel):
    pgn: str
    model: str = "maia3-5m"
    elo: int = 1500
    multipv: int = 5


class HumanReviewSummary(BaseModel):
    human_match_percent: float = 0.0
    most_human_side: str = ""
    sharpest_moments: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class HumanReviewMove(BaseModel):
    ply: int
    move: str
    move_uci: str
    fen: str
    last_move: list[str] = Field(default_factory=list)
    played_probability: float = 0.0
    typicality: str = ""
    human_label: str = ""
    candidates: list[MoveCandidate] = Field(default_factory=list)


class HumanReviewReport(BaseModel):
    version: int = 1
    source: str = "maia3_microservice"
    model: str = "maia3-5m"
    elo: int = 1500
    generated_at: str
    summary: HumanReviewSummary
    moves: list[HumanReviewMove] = Field(default_factory=list)
