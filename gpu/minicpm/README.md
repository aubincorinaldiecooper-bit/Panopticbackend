# MiniCPM-V 4.6 watcher server

The model that watches internet videos for Clipit's live search: MiniCPM-V 4.6
(about 1.3 billion parameters, 2.6 GB of weights), served by vLLM on one GPU
server. Several watchers share the GPU: vLLM batches their requests.

Nothing here has been run on a GPU yet. The engine side is tested against a
stand-in; the first real run is the measured test on Vultr.

## Pinned versions

- vLLM `v0.34.0`, the image `vllm/vllm-openai:v0.34.0`. MiniCPM-V 4.6 support
  landed in vLLM on 2026-05-22 (vllm-project/vllm#43213); v0.34.0 contains it.
- Model `openbmb/MiniCPM-V-4.6` at revision
  `97328f3c2d70c4645b6a35594807eb6932068635` (published 2026-11-11).
- Caddy `2.8.4`.

## Security

vLLM's `--api-key` (here `VLLM_API_KEY`) protects only routes under `/v1`.
Other routes on the same server, including `/invocations`, run the model
without a key. So:

1. vLLM is never published; only Caddy is, and it forwards `/v1/*` only.
2. The server's firewall should allow 443 only from the Clipit engine's
   server (and 80 while Caddy obtains its certificate), plus SSH from where
   you manage it.

## Running it

On an Ubuntu GPU server with the NVIDIA driver and the NVIDIA container
toolkit installed (`nvidia-smi` should list the GPU, and
`docker run --rm --gpus all ubuntu nvidia-smi` should too):

1. Point a DNS name at the server, e.g. `minicpm.example.com`.
2. Copy this folder to the server, then `cp .env.example .env` and fill both values.
3. `docker compose up -d`. The first start downloads the model (2.6 GB).
4. Check it answers, with the key, and only with the key:

   ```bash
   curl -s https://$MINICPM_DOMAIN/v1/models -H "Authorization: Bearer $MINICPM_API_KEY"
   curl -s -o /dev/null -w '%{http_code}\n' https://$MINICPM_DOMAIN/v1/models
   curl -s -o /dev/null -w '%{http_code}\n' https://$MINICPM_DOMAIN/invocations
   ```

   The first lists `openbmb/MiniCPM-V-4.6`; the second answers 401; the third 404.

## Pointing the engine at it

On the Clipit worker:

```text
INTERNET_WATCHER=minicpm
MINICPM_WATCHER_URL=https://minicpm.example.com/v1
MINICPM_WATCHER_API_KEY=<the same MINICPM_API_KEY>
MINICPM_WATCHER_MODEL=openbmb/MiniCPM-V-4.6
MINICPM_WATCHER_REVISION=97328f3c2d70c4645b6a35594807eb6932068635
```

How the engine uses it: each scout groups the browser's frames into windows
of source time (`MINICPM_WATCH_WINDOW_SECONDS`, default 8), sends up to
`MINICPM_WATCH_FRAMES_PER_WINDOW` (default 8) evenly spread frames per window
with the search, and reads back the matching moments as JSON, validated and
held to the window. A window the model could not judge, or that was skipped
because the watcher fell behind the player, is recorded as unexamined.
