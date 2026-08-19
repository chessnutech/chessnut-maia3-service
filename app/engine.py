from __future__ import annotations

# SPDX-License-Identifier: AGPL-3.0-only
#
# AGPL-3.0 compliance boundary: this file may wrap Maia 3, PyTorch, python-chess,
# and model weights. Do not import it from the closed-source Chessnut app or
# main Go backend; use the HTTP API exposed by app.main instead.

import os
import queue
import random
import re
import shlex
import shutil
import sys
import threading
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from .schemas import (
    BotMoveRequest,
    BotMoveResponse,
    HumanReviewMove,
    HumanReviewReport,
    HumanReviewSummary,
    MoveCandidate,
    ReviewGameRequest,
)

MANAGED_UCI_OPTIONS = {"multipv", "ponder", "uci_chess960", "uci_variant"}
DEFAULT_MAIA3_MODE = "uci"
MAIA3_BOT_POOL = "bot"
MAIA3_REVIEW_POOL = "review"


def _optional_pool_size(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


@dataclass(frozen=True)
class Maia3Settings:
    mode: str = os.getenv("MAIA3_MODE", DEFAULT_MAIA3_MODE)
    command: str = os.getenv(
        "MAIA3_COMMAND",
        "maia3-uci --model 5m --use-uci-history --device cpu --no-use-amp",
    )
    native_session_pool_size: int = int(
        os.getenv("MAIA3_SESSION_POOL_SIZE", "2") or "2"
    )
    bot_session_pool_size: int | None = _optional_pool_size(
        "MAIA3_BOT_SESSION_POOL_SIZE"
    )
    review_session_pool_size: int | None = _optional_pool_size(
        "MAIA3_REVIEW_SESSION_POOL_SIZE"
    )


class Maia3Engine:
    """AGPL-3.0 boundary wrapper for Maia3 inference.

    This code belongs to the independently deployed Maia3 microservice. Do not
    import it from the closed-source Chessnut app or main Go backend.
    """

    def __init__(self, settings: Maia3Settings | None = None) -> None:
        self.settings = settings or Maia3Settings()
        self._native_session_cache: dict[tuple[str, ...], _NativeMaia3SessionPool] = {}
        self._native_session_cache_lock = threading.Lock()

    def bot_move(self, request: BotMoveRequest) -> BotMoveResponse:
        board = _board_from_request(request)
        return self._bot_move_for_board(board, request)

    def review_game(self, request: ReviewGameRequest) -> HumanReviewReport:
        game = chess.pgn.read_game(__import__("io").StringIO(request.pgn))
        if game is None:
            raise ValueError("PGN could not be read.")
        board = game.board()
        moves: list[HumanReviewMove] = []
        history: list[str] = []
        matches = 0
        with self._review_context(request) as review_engine:
            for ply, move in enumerate(game.mainline_moves(), start=1):
                fen_before = board.fen()
                san = board.san(move)
                candidate_response = self._review_bot_move(
                    review_engine,
                    board,
                    BotMoveRequest(
                        fen=fen_before,
                        moves=history,
                        model=request.model,
                        elo=request.elo,
                        multipv=request.multipv,
                    ),
                )
                played_probability = _candidate_probability(
                    candidate_response.candidates,
                    move.uci(),
                )
                if played_probability == 0.0 and hasattr(
                    review_engine,
                    "move_probability",
                ):
                    played_probability = review_engine.move_probability(
                        board,
                        BotMoveRequest(
                            fen=fen_before,
                            moves=history,
                            model=request.model,
                            elo=request.elo,
                            multipv=request.multipv,
                        ),
                        move.uci(),
                    )
                if played_probability >= 0.18:
                    matches += 1
                history.append(move.uci())
                board.push(move)
                moves.append(
                    HumanReviewMove(
                        ply=ply,
                        move=san,
                        move_uci=move.uci(),
                        fen=board.fen(),
                        last_move=[
                            chess.square_name(move.from_square),
                            chess.square_name(move.to_square),
                        ],
                        played_probability=played_probability,
                        typicality=_typicality(played_probability),
                        human_label=_human_label(played_probability),
                        candidates=candidate_response.candidates,
                    )
                )
        total = max(1, len(moves))
        white_match = _side_match(moves, odd=True)
        black_match = _side_match(moves, odd=False)
        return HumanReviewReport(
            model=request.model,
            elo=request.elo,
            generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            summary=HumanReviewSummary(
                human_match_percent=round(matches * 100 / total, 1),
                most_human_side="white" if white_match >= black_match else "black",
                sharpest_moments=_sharpest_moments(moves),
                notes=[
                    "Maia3 shows likely human choices, not Stockfish best moves.",
                ],
            ),
            moves=moves,
        )

    def _review_context(self, request: ReviewGameRequest):
        if self.settings.mode != "uci":
            return _NullReviewContext()
        command = resolve_maia3_command(
            command_for_maia3_model(
                parse_maia3_command(self.settings.command),
                request.model,
            )
        )
        native_session = self._native_maia3_session(
            command,
            pool_name=MAIA3_REVIEW_POOL,
        )
        if native_session is not None:
            return native_session
        engine = chess.engine.SimpleEngine.popen_uci(command)
        config = build_uci_config(
            engine.options,
            BotMoveRequest(
                fen=chess.STARTING_FEN,
                model=request.model,
                elo=request.elo,
                multipv=request.multipv,
            ),
        )
        if config:
            engine.configure(config)
        return engine

    def _review_bot_move(
        self,
        review_engine: object,
        board: chess.Board,
        request: BotMoveRequest,
    ) -> BotMoveResponse:
        if self.settings.mode == "uci":
            if hasattr(review_engine, "score"):
                return self._native_bot_move_with_session(review_engine, board, request)
            return self._uci_bot_move_with_engine(review_engine, board, request)
        return self._mock_bot_move(board, request)

    def _bot_move_for_board(
        self,
        board: chess.Board,
        request: BotMoveRequest,
    ) -> BotMoveResponse:
        if self.settings.mode == "uci":
            return self._uci_bot_move(board, request)
        return self._mock_bot_move(board, request)

    def _mock_bot_move(
        self,
        board: chess.Board,
        request: BotMoveRequest,
    ) -> BotMoveResponse:
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("No legal moves in this position.")
        seed = hash((request.fen, tuple(request.moves), request.elo, request.model))
        rng = random.Random(seed)
        ranked = sorted(legal, key=lambda move: _move_shape_score(board, move), reverse=True)
        pool = ranked[: max(1, min(len(ranked), request.multipv or 5))]
        selected = pool[rng.randrange(len(pool))]
        candidates = _mock_candidates(board, pool)
        return BotMoveResponse(
            move=selected.uci(),
            san=board.san(selected),
            probability=candidates[0].probability if candidates else 0.0,
            model=request.model,
            elo=request.elo,
            candidates=candidates,
        )

    def _uci_bot_move(
        self,
        board: chess.Board,
        request: BotMoveRequest,
    ) -> BotMoveResponse:
        command = resolve_maia3_command(
            command_for_maia3_model(
                parse_maia3_command(self.settings.command),
                request.model,
            )
        )
        native_session = self._native_maia3_session(
            command,
            pool_name=MAIA3_BOT_POOL,
        )
        if native_session is not None:
            with native_session as session:
                return self._native_bot_move_with_session(session, board, request)
        with chess.engine.SimpleEngine.popen_uci(command) as engine:
            config = build_uci_config(engine.options, request)
            if config:
                engine.configure(config)
            return self._uci_bot_move_with_engine(engine, board, request)

    def _uci_bot_move_with_engine(
        self,
        engine: object,
        board: chess.Board,
        request: BotMoveRequest,
    ) -> BotMoveResponse:
        limit = chess.engine.Limit(nodes=1)
        info = engine.analyse(board, limit, multipv=max(1, request.multipv))
        if isinstance(info, dict):
            info = [info]
        candidates: list[MoveCandidate] = []
        for item in info:
            pv = item.get("pv") or []
            if not pv:
                continue
            move = pv[0]
            candidates.append(
                MoveCandidate(
                    move=move.uci(),
                    san=board.san(move),
                    probability=_probability_from_info(item),
                    wdl=_wdl_from_info(item),
                    centipawns=_cp_from_info(item),
                )
            )
        selected = candidates[0] if candidates else None
        if selected is None:
            result = engine.play(board, limit)
            selected = MoveCandidate(
                move=result.move.uci(),
                san=board.san(result.move),
                probability=0.0,
            )
        return BotMoveResponse(
            move=selected.move,
            san=selected.san,
            probability=selected.probability,
            model=request.model,
            elo=request.elo,
            candidates=candidates or [selected],
        )

    def _native_bot_move_with_session(
        self,
        session: "_NativeMaia3Session",
        board: chess.Board,
        request: BotMoveRequest,
    ) -> BotMoveResponse:
        selected_move, top_moves = session.score(board, request)
        candidates: list[MoveCandidate] = []
        for item in top_moves:
            move = item.get("move")
            if not isinstance(move, chess.Move):
                continue
            candidates.append(
                MoveCandidate(
                    move=move.uci(),
                    san=board.san(move),
                    probability=_normalize_probability(item.get("policy")),
                    wdl=_wdl_from_native_item(item),
                    centipawns=_cp_from_native_item(item),
                )
            )

        if selected_move is None and candidates:
            selected_move = chess.Move.from_uci(candidates[0].move)
        if selected_move is None:
            raise ValueError("No legal moves in this position.")

        selected_probability = _candidate_probability(candidates, selected_move.uci())
        if selected_probability == 0.0 and hasattr(session, "move_probability"):
            selected_probability = session.move_probability(
                board,
                request,
                selected_move.uci(),
            )
        return BotMoveResponse(
            move=selected_move.uci(),
            san=board.san(selected_move),
            probability=selected_probability,
            model=request.model,
            elo=request.elo,
            candidates=candidates,
        )

    def _native_maia3_session(
        self,
        command: list[str],
        *,
        pool_name: str,
    ) -> "_NativeMaia3SessionLease | None":
        command_key = _native_maia3_session_key(command)
        if not command or command_key is None:
            return None
        key = (pool_name, *command_key)
        with self._native_session_cache_lock:
            pool = self._native_session_cache.get(key)
            if pool is None:
                try:
                    pool = _NativeMaia3SessionPool(
                        command,
                        self._pool_size(pool_name),
                    )
                except Exception:
                    return None
                self._native_session_cache[key] = pool
            return pool.lease()

    def _pool_size(self, pool_name: str) -> int:
        if pool_name == MAIA3_BOT_POOL:
            configured = self.settings.bot_session_pool_size
        elif pool_name == MAIA3_REVIEW_POOL:
            configured = self.settings.review_session_pool_size
        else:
            raise ValueError(f"Unknown Maia3 session pool: {pool_name}")
        return configured or self.settings.native_session_pool_size


class _NullReviewContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


class _NativeMaia3SessionPool:
    """Small lazy pool for native Maia3 sessions.

    AGPL-3.0 compliance boundary: this pool only lives inside the open-source
    Maia3 microservice and prevents one cached model session from serializing
    every user request.
    """

    def __init__(self, command: list[str], size: int) -> None:
        self._command = command
        self._size = max(1, size)
        self._created = 0
        self._lock = threading.Lock()
        self._available: queue.LifoQueue[_NativeMaia3Session] = queue.LifoQueue(
            maxsize=self._size,
        )

    def lease(self) -> "_NativeMaia3SessionLease":
        return _NativeMaia3SessionLease(self)

    def acquire(self) -> "_NativeMaia3Session":
        try:
            return self._available.get_nowait()
        except queue.Empty:
            pass

        with self._lock:
            if self._created < self._size:
                self._created += 1
                create = True
            else:
                create = False

        if create:
            try:
                return _NativeMaia3Session(self._command)
            except Exception:
                with self._lock:
                    self._created -= 1
                raise

        return self._available.get()

    def release(self, session: "_NativeMaia3Session") -> None:
        self._available.put(session)


class _NativeMaia3SessionLease:
    def __init__(self, pool: _NativeMaia3SessionPool) -> None:
        self._pool = pool
        self._session: _NativeMaia3Session | None = None

    def __enter__(self) -> "_NativeMaia3Session":
        self._session = self._pool.acquire()
        return self._session

    def __exit__(self, *args: object) -> None:
        if self._session is not None:
            self._pool.release(self._session)
            self._session = None


class _NativeMaia3Session:
    """Direct Maia3 package adapter for real policy probabilities.

    AGPL-3.0 compliance boundary: this adapter imports Maia3 inside the
    independently deployed microservice only. The closed-source app/backend
    still communicate with this service over HTTP.
    """

    options: dict[str, object] = {}

    def __init__(self, command: list[str]) -> None:
        from maia3.uci import Maia3UCIEngine, clamp_multipv, parse_args as parse_uci_args

        self._clamp_multipv = clamp_multipv
        self._engine = Maia3UCIEngine(parse_uci_args(command[1:]))
        self._lock = threading.Lock()

    def __enter__(self) -> "_NativeMaia3Session":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def score(
        self,
        board: chess.Board,
        request: BotMoveRequest,
    ) -> tuple[chess.Move | None, list[dict[str, object]]]:
        with self._lock:
            engine = self._engine
            engine.self_elo = request.elo
            engine.oppo_elo = request.elo
            engine.temperature = request.temperature
            engine.top_p = request.top_p
            engine.multipv = self._clamp_multipv(request.multipv)
            self._set_position(board, request.moves)
            engine.ensure_model_loaded()
            return engine.score_moves()

    def move_probability(
        self,
        board: chess.Board,
        request: BotMoveRequest,
        move_uci: str,
    ) -> float:
        from maia3.dataset import get_legal_moves_mask
        from maia3.utils import mirror_move
        import torch
        from torch.amp import autocast

        with self._lock:
            engine = self._engine
            engine.self_elo = request.elo
            engine.oppo_elo = request.elo
            engine.temperature = request.temperature
            engine.top_p = request.top_p
            self._set_position(board, request.moves)
            engine.ensure_model_loaded()
            legal_mask = get_legal_moves_mask(engine.board, engine.all_moves_dict)
            if not bool(legal_mask.any()):
                return 0.0
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                return 0.0
            if move not in engine.board.legal_moves:
                return 0.0
            model_move_uci = move.uci()
            if engine.board.turn == chess.BLACK:
                model_move_uci = mirror_move(model_move_uci)
            move_index = engine.all_moves_dict.get(model_move_uci)
            if move_index is None:
                return 0.0
            tokens = engine._tokens_from_history(engine.history)
            tokens = tokens.unsqueeze(0).to(engine.cfg.device)
            self_elos = torch.tensor(
                [engine.self_elo],
                dtype=torch.long,
                device=engine.cfg.device,
            )
            oppo_elos = torch.tensor(
                [engine.oppo_elo],
                dtype=torch.long,
                device=engine.cfg.device,
            )
            with autocast(
                "cuda",
                enabled=engine.cfg.use_amp and engine.cfg.device.startswith("cuda"),
            ):
                logits_move, _logits_value, _ = engine.model(
                    tokens,
                    self_elos,
                    oppo_elos,
                )
            logits = logits_move[0].float()
            mask = legal_mask.to(engine.cfg.device)
            logits = logits.masked_fill(~mask, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            return _normalize_probability(probabilities[move_index].item())

    def _set_position(self, board: chess.Board, requested_moves: list[str]) -> None:
        history_moves = [move.strip() for move in requested_moves if move.strip()]
        if not history_moves and board.move_stack:
            history_moves = [move.uci() for move in board.move_stack]

        if history_moves:
            self._engine.cmd_position("position startpos moves " + " ".join(history_moves))
            if _same_position(self._engine.board, board):
                return

        self._engine.cmd_position("position fen " + board.fen())


def _native_maia3_session_key(command: list[str]) -> tuple[str, ...] | None:
    if not command:
        return None
    executable = Path(command[0]).name.lower()
    if executable not in {"maia3-uci", "maia3-uci.exe"}:
        return None
    return tuple(command)


def parse_maia3_command(command: str) -> list[str]:
    """Parse MAIA3_COMMAND for the AGPL-3.0 Maia 3 microservice boundary."""

    return shlex.split(command, posix=True)


def command_for_maia3_model(command: list[str], model_name: str) -> list[str]:
    if not command:
        return command
    model_alias = _maia3_model_alias(model_name)
    updated = list(command)
    for index, token in enumerate(updated):
        if token == "--model" and index + 1 < len(updated):
            updated[index + 1] = model_alias
            return updated
        if token.startswith("--model="):
            updated[index] = f"--model={model_alias}"
            return updated
    return [*updated, "--model", model_alias]


def _maia3_model_alias(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if normalized in {"maia3-23m", "23m"}:
        return "23m"
    if normalized in {"maia3-79m", "79m"}:
        return "79m"
    if normalized in {"maia3-3m", "maia3-3m-ablation", "3m"}:
        return "3m"
    return "5m"


def resolve_maia3_command(command: list[str]) -> list[str]:
    if not command:
        return command
    executable = command[0]
    if shutil.which(executable):
        return command
    if os.path.isabs(executable) or os.path.dirname(executable):
        return command

    for bin_name in ("Scripts", "bin"):
        suffixes = ("", ".exe") if os.name == "nt" else ("",)
        for suffix in suffixes:
            candidate = Path(sys.prefix) / bin_name / f"{executable}{suffix}"
            if candidate.exists():
                return [str(candidate), *command[1:]]
    return command


def build_uci_config(
    engine_options: dict[str, object],
    request: BotMoveRequest,
) -> dict[str, object]:
    """Build safe Maia3 UCI options.

    python-chess manages options such as MultiPV internally through analyse().
    Sending those through configure() raises an EngineError in real UCI mode.
    """

    desired: dict[str, object] = {
        "Elo": request.elo,
        "Temperature": request.temperature,
        "TopP": request.top_p,
    }
    available = {name.lower(): name for name in engine_options}
    config: dict[str, object] = {}
    for requested_name, value in desired.items():
        actual_name = available.get(requested_name.lower())
        if actual_name is None or actual_name.lower() in MANAGED_UCI_OPTIONS:
            continue
        config[actual_name] = value
    return config


def _board_from_request(request: BotMoveRequest) -> chess.Board:
    fen_board = chess.Board(request.fen)
    history_board = chess.Board()
    used_history = False
    try:
        for uci in request.moves:
            move_text = uci.strip()
            if not move_text:
                continue
            move = chess.Move.from_uci(move_text)
            if move not in history_board.legal_moves:
                raise ValueError(f"illegal history move: {move_text}")
            history_board.push(move)
            used_history = True
        if used_history and _same_position(history_board, fen_board):
            return history_board
        if used_history and request.fen.strip() == chess.STARTING_FEN:
            return history_board
    except ValueError:
        pass
    return fen_board


def _same_position(left: chess.Board, right: chess.Board) -> bool:
    return (
        left.board_fen() == right.board_fen()
        and left.turn == right.turn
        and left.castling_rights == right.castling_rights
        and left.ep_square == right.ep_square
    )


def _candidate_probability(candidates: list[MoveCandidate], uci: str) -> float:
    for candidate in candidates:
        if candidate.move == uci:
            return candidate.probability
    return 0.0


def _typicality(probability: float) -> str:
    if probability >= 0.25:
        return "common"
    if probability >= 0.10:
        return "plausible"
    if probability > 0:
        return "rare"
    return "unseen"


def _human_label(probability: float) -> str:
    if probability >= 0.25:
        return "Common human choice"
    if probability >= 0.10:
        return "Plausible human choice"
    if probability > 0:
        return "Rare human choice"
    return "Unusual compared with Maia3 candidates"


def _move_shape_score(board: chess.Board, move: chess.Move) -> int:
    score = 0
    if board.is_capture(move):
        score += 40
    if board.gives_check(move):
        score += 25
    if move.promotion:
        score += 60
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    score += 12 - abs(to_file - 3) - abs(to_rank - 3)
    return score


def _mock_candidates(board: chess.Board, moves: list[chess.Move]) -> list[MoveCandidate]:
    weights = [max(1, _move_shape_score(board, move) + 50) for move in moves]
    total = sum(weights)
    return [
        MoveCandidate(
            move=move.uci(),
            san=board.san(move),
            probability=round(weight / total, 4),
            wdl=[350 + index * 4, 340, 310 - index * 4],
            centipawns=12 - index * 3,
        )
        for index, (move, weight) in enumerate(zip(moves, weights))
    ]


def _side_match(moves: list[HumanReviewMove], odd: bool) -> float:
    selected = [m.played_probability for m in moves if (m.ply % 2 == 1) == odd]
    if not selected:
        return 0.0
    return sum(selected) / len(selected)


def _sharpest_moments(moves: list[HumanReviewMove]) -> list[int]:
    ranked = sorted(moves, key=lambda move: move.played_probability)
    return [move.ply for move in ranked[:3]]


def _probability_from_info(info: dict) -> float:
    direct = _normalize_probability(info.get("policy"))
    if direct > 0:
        return direct
    text = str(info.get("string") or "")
    match = re.search(r"(?:policy|probability|prob)\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.I)
    if match:
        return _normalize_probability(match.group(1))
    return 0.0


def _normalize_probability(value: object) -> float:
    try:
        probability = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if probability > 1.0 and probability <= 100.0:
        probability = probability / 100.0
    if probability < 0.0:
        return 0.0
    if probability > 1.0:
        return 1.0
    return round(probability, 4)


def _wdl_from_native_item(item: dict[str, object]) -> list[int]:
    wdl = item.get("wdl")
    if not isinstance(wdl, tuple) and not isinstance(wdl, list):
        return []
    return [int(value) for value in list(wdl)[:3]]


def _cp_from_native_item(item: dict[str, object]) -> int:
    wdl = _wdl_from_native_item(item)
    if len(wdl) == 3:
        return int(wdl[0] - wdl[2])
    return 0


def _wdl_from_info(info: dict) -> list[int]:
    wdl = info.get("wdl")
    if wdl is None:
        return []
    return wdl_values(wdl)


def wdl_values(wdl: object) -> list[int]:
    if isinstance(wdl, chess.engine.PovWdl):
        wdl = wdl.white()
    if isinstance(wdl, chess.engine.Wdl):
        return [int(wdl.wins), int(wdl.draws), int(wdl.losses)]
    try:
        values = list(wdl)  # type: ignore[arg-type]
    except TypeError:
        return []
    return [int(value) for value in values[:3]]


def _cp_from_info(info: dict) -> int:
    score = info.get("score")
    if score is None:
        return 0
    pov = score.white()
    return int(pov.score(mate_score=100000) or 0)
