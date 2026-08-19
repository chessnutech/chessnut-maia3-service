from __future__ import annotations

# SPDX-License-Identifier: AGPL-3.0-only
#
# AGPL-3.0 compliance boundary: this file belongs to the independently
# deployable Maia 3 inference microservice. The closed-source Chessnut app and
# main backend may only communicate with this service through HTTP APIs.

from fastapi import FastAPI, HTTPException

from .engine import Maia3Engine
from .schemas import BotMoveRequest, BotMoveResponse, HumanReviewReport, ReviewGameRequest

app = FastAPI(
    title="Chessnut Maia3 Inference Microservice",
    version="0.1.0",
    description=(
        "AGPL-3.0 open-source boundary. The closed-source Chessnut app and "
        "main backend communicate with this service only over HTTP."
    ),
)
engine = Maia3Engine()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "maia3_microservice"}


@app.post("/v1/bot/move", response_model=BotMoveResponse)
def bot_move(request: BotMoveRequest) -> BotMoveResponse:
    try:
        return engine.bot_move(request)
    except Exception as exc:  # pragma: no cover - converted to API response
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/v1/review/game", response_model=HumanReviewReport)
def review_game(request: ReviewGameRequest) -> HumanReviewReport:
    try:
        return engine.review_game(request)
    except Exception as exc:  # pragma: no cover - converted to API response
        raise HTTPException(status_code=503, detail=str(exc)) from exc
