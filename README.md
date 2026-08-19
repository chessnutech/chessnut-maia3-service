# Chessnut Maia3 Inference Microservice

This directory is the AGPL-3.0 open-source boundary for Maia3 integration.

The closed-source Chessnut app and main Go backend must not import Maia3,
PyTorch, or this Python package directly. They may only communicate with this
service through HTTP.

## Compliance Boundary

- Closed source: Flutter apps, main Go backend, user data, wallet, membership,
  game records, and product logic.
- Open source under AGPL-3.0: this Maia3 inference service, its API wrapper,
  and any Maia3 engine modifications used here.

Maia3 upstream: <https://github.com/CSSLab/maia3>

The app should expose the source-code URL for this microservice in legal/open
source notices before production release.

## API

- `GET /health`
- `POST /v1/bot/move`
- `POST /v1/review/game`

The service defaults to real UCI mode. Install Maia3 from its official GitHub
repository before starting the service. Mock mode is only intended for an
explicitly configured development or staging smoke test (`MAIA3_MODE=mock`).

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -i https://pypi.org/simple "git+https://github.com/CSSLab/maia3.git"
.\.venv\Scripts\maia3-uci --list-models
```

The first real inference may download Maia3 weights from Hugging Face. For a
warm cache, run a one-move UCI smoke test after deployment:

```powershell
@"
uci
isready
position startpos moves e2e4
go nodes 1
quit
"@ | .\.venv\Scripts\maia3-uci --model 5m --use-uci-history --device cpu --no-use-amp
```

Then run the service. `MAIA3_MODE` may be set explicitly, but defaults to
`uci` when omitted:

```powershell
$env:MAIA3_MODE="uci"
$env:MAIA3_COMMAND="maia3-uci --model 5m --use-uci-history --device cpu --no-use-amp"
$env:MAIA3_BOT_SESSION_POOL_SIZE="2"
$env:MAIA3_REVIEW_SESSION_POOL_SIZE="2"
```

`MAIA3_BOT_SESSION_POOL_SIZE` and `MAIA3_REVIEW_SESSION_POOL_SIZE` control
independent native Maia3 pools per model command. Both default to `2`, so a
whole-game review never holds a session needed by a live bot move. The legacy
`MAIA3_SESSION_POOL_SIZE` remains the fallback for either value when its
specific variable is omitted. Match the Go global request guards to these
pool sizes and size both pools to the CPU/GPU/RAM available on the inference
host.

## AGPL Notice

This microservice is intended to be published under GNU AGPL-3.0. Keep this
directory independently deployable and independently publishable.
