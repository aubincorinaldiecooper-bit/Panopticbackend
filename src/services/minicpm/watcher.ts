import { z } from 'zod';
import { env } from '../../config/env.js';
import { ExternalServiceError } from '../../lib/errors.js';
import { parseModelJson } from '../search/modelResponse.js';

/**
 * MiniCPM-V 4.6 as the live search's watcher, served by vLLM on the team's
 * own GPU server and reached over its OpenAI-compatible API.
 *
 * One call judges one WINDOW: a few seconds of a browser's playback, as a
 * handful of timestamped frames sent together, plus the search. The model
 * answers with the moments in that window that show what was asked for, as
 * JSON — which is untrusted input like everything a model returns: it is
 * parsed, validated and held inside the window's own time range here, before
 * anything above this module sees it.
 *
 * A window that could not be judged (the server failed, timed out, or
 * answered with something that is not the expected JSON) throws. The caller
 * records it as unexamined. It is never turned into "nothing was there".
 */

/**
 * MiniCPM-V 4.6 ends its turn with these (it uses the Qwen3.5 vocabulary);
 * the model's vLLM guide requires them on every request.
 */
export const MINICPM_V46_STOP_TOKEN_IDS = [248044, 248046] as const;

export interface WindowFrame {
  /** Position in the source video, not wall-clock time. */
  timestampMs: number;
  durationMs: number;
  encoding: 'jpeg' | 'png';
  image: Buffer;
}

export interface WindowMoment {
  startSeconds: number;
  endSeconds: number;
  description: string;
  /** How sure the watcher said it was, 0 to 1. Absent when it did not say, or said something unusable. */
  confidence?: number;
}

export interface WindowJudgement {
  moments: WindowMoment[];
  /** Matches the model placed wholly outside the window it was shown, and so dropped. */
  outsideWindow: number;
  /** The reply needed the known fullwidth-colon repair to parse. */
  repaired: boolean;
  latencyMs: number;
  promptTokens?: number;
  completionTokens?: number;
}

interface WatcherConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
  timeoutMs: number;
}

function watcherConfig(): WatcherConfig {
  if (!env.MINICPM_WATCHER_URL || !env.MINICPM_WATCHER_API_KEY) {
    throw new ExternalServiceError(
      'minicpm-watcher',
      'The MiniCPM watcher is not configured: set MINICPM_WATCHER_URL and MINICPM_WATCHER_API_KEY',
      { retryable: false },
    );
  }
  return {
    baseUrl: env.MINICPM_WATCHER_URL.replace(/\/+$/, ''),
    apiKey: env.MINICPM_WATCHER_API_KEY,
    model: env.MINICPM_WATCHER_MODEL,
    timeoutMs: env.MINICPM_WATCH_TIMEOUT_SECONDS * 1000,
  };
}

function seconds(ms: number): string {
  return (Math.round(ms / 100) / 10).toFixed(1);
}

/** The request for one window. Exported so tests can pin its shape. */
export function buildWindowRequest(input: { frames: WindowFrame[]; query: string; model: string }): Record<string, unknown> {
  if (input.frames.length === 0) throw new ExternalServiceError('minicpm-watcher', 'a window needs at least one frame', { retryable: false });
  const first = input.frames[0]!;
  const last = input.frames[input.frames.length - 1]!;
  const from = seconds(first.timestampMs);
  const to = seconds(last.timestampMs + last.durationMs);
  const content: Array<Record<string, unknown>> = [];
  for (const frame of input.frames) {
    content.push({ type: 'text', text: `Frame at ${seconds(frame.timestampMs)}s:` });
    content.push({ type: 'image_url', image_url: { url: `data:image/${frame.encoding};base64,${frame.image.toString('base64')}` } });
  }
  content.push({
    type: 'text',
    text: [
      `Search: ${input.query}`,
      '',
      `These frames run from ${from}s to ${to}s of one video, in order. Does anything in them show what the search asks for?`,
      'Answer with JSON only, in exactly this shape:',
      '{"matches": [{"start": <seconds>, "end": <seconds>, "description": "<what is visible>", "confidence": <0 to 1>}]}',
      `Use only times between ${from} and ${to}. Describe only what is visible. If nothing in these frames shows it, answer {"matches": []}.`,
    ].join('\n'),
  });
  return {
    model: input.model,
    messages: [
      { role: 'system', content: 'You watch short stretches of video for a search, and report only what is visible in the frames you are given.' },
      { role: 'user', content },
    ],
    temperature: 0,
    max_tokens: 400,
    stop_token_ids: [...MINICPM_V46_STOP_TOKEN_IDS],
  };
}

const replySchema = z.object({
  matches: z.array(z.unknown()),
});

const matchSchema = z.object({
  start: z.number().finite(),
  end: z.number().finite(),
  description: z.string().optional(),
  confidence: z.unknown().optional(),
});

function sureness(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
  if (value < 0 || value > 1) return undefined;
  return value;
}

/**
 * Validate what the model said against the window it was shown.
 *
 * A match must be a well-formed interval. One that lies wholly outside the
 * window is dropped and counted (the model cannot have seen it); one that
 * overhangs is held to the window's edges. A single-instant match is given
 * the span of the frame it names, so it is never a zero-length moment.
 */
export function readWindowReply(
  rawText: string,
  window: { startMs: number; endMs: number; frameMs: number },
): { moments: WindowMoment[]; outsideWindow: number; repaired: boolean } {
  const parsed = parseModelJson(rawText);
  if (!parsed.ok) {
    throw new ExternalServiceError('minicpm-watcher', `MiniCPM reply was not JSON (${parsed.stage})`, { retryable: false });
  }
  const reply = replySchema.safeParse(parsed.value);
  if (!reply.success) {
    throw new ExternalServiceError('minicpm-watcher', 'MiniCPM reply did not have a "matches" list', { retryable: false });
  }
  const windowStart = window.startMs / 1000;
  const windowEnd = window.endMs / 1000;
  const moments: WindowMoment[] = [];
  let outsideWindow = 0;
  for (const row of reply.data.matches) {
    const match = matchSchema.safeParse(row);
    if (!match.success) {
      throw new ExternalServiceError('minicpm-watcher', 'MiniCPM reply had a malformed match', { retryable: false });
    }
    let start = Math.min(match.data.start, match.data.end);
    let end = Math.max(match.data.start, match.data.end);
    if (end < windowStart || start > windowEnd) {
      outsideWindow += 1;
      continue;
    }
    start = Math.max(windowStart, start);
    end = Math.min(windowEnd, end);
    if (end <= start) end = Math.min(windowEnd, start + Math.max(0.1, window.frameMs / 1000));
    if (end <= start) start = Math.max(windowStart, end - Math.max(0.1, window.frameMs / 1000));
    const confidence = sureness(match.data.confidence);
    moments.push({
      startSeconds: Math.round(start * 1000) / 1000,
      endSeconds: Math.round(end * 1000) / 1000,
      description: (match.data.description ?? '').trim().slice(0, 1000),
      ...(confidence === undefined ? {} : { confidence }),
    });
  }
  return { moments, outsideWindow, repaired: parsed.repaired };
}

/** Judge one window of frames. Throws when the window could not be judged. */
export async function judgeWindow(input: { frames: WindowFrame[]; query: string; signal?: AbortSignal }): Promise<WindowJudgement> {
  const config = watcherConfig();
  const body = buildWindowRequest({ frames: input.frames, query: input.query, model: config.model });
  const first = input.frames[0]!;
  const last = input.frames[input.frames.length - 1]!;
  const startedAt = Date.now();
  const signal = input.signal
    ? AbortSignal.any([input.signal, AbortSignal.timeout(config.timeoutMs)])
    : AbortSignal.timeout(config.timeoutMs);

  let response: Response;
  try {
    response = await fetch(`${config.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { authorization: `Bearer ${config.apiKey}`, 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    throw new ExternalServiceError('minicpm-watcher', `MiniCPM watcher unreachable: ${error instanceof Error ? error.message : String(error)}`, { retryable: true });
  }
  if (!response.ok) {
    const detail = (await response.text().catch(() => '')).slice(0, 300);
    throw new ExternalServiceError('minicpm-watcher', `MiniCPM watcher answered ${response.status}${detail ? `: ${detail}` : ''}`, {
      retryable: response.status === 429 || response.status >= 500,
    });
  }
  const payload = (await response.json().catch(() => null)) as {
    choices?: Array<{ message?: { content?: unknown } }>;
    usage?: { prompt_tokens?: unknown; completion_tokens?: unknown };
  } | null;
  const text = payload?.choices?.[0]?.message?.content;
  if (typeof text !== 'string') {
    throw new ExternalServiceError('minicpm-watcher', 'MiniCPM watcher returned no message text', { retryable: false });
  }
  const read = readWindowReply(text, {
    startMs: first.timestampMs,
    endMs: last.timestampMs + last.durationMs,
    frameMs: last.durationMs,
  });
  const promptTokens = typeof payload?.usage?.prompt_tokens === 'number' ? payload.usage.prompt_tokens : undefined;
  const completionTokens = typeof payload?.usage?.completion_tokens === 'number' ? payload.usage.completion_tokens : undefined;
  return {
    ...read,
    latencyMs: Date.now() - startedAt,
    ...(promptTokens === undefined ? {} : { promptTokens }),
    ...(completionTokens === undefined ? {} : { completionTokens }),
  };
}

/**
 * Check the server is up and serves the model we are about to ask, before a
 * search spends anything on browsers. Cheap: one small request, no GPU work.
 */
export async function assertMiniCpmWatcherReady(): Promise<void> {
  const config = watcherConfig();
  let response: Response;
  try {
    response = await fetch(`${config.baseUrl}/models`, {
      headers: { authorization: `Bearer ${config.apiKey}` },
      signal: AbortSignal.timeout(10_000),
    });
  } catch (error) {
    throw new ExternalServiceError('minicpm-watcher', `MiniCPM watcher unreachable: ${error instanceof Error ? error.message : String(error)}`, { retryable: true });
  }
  if (!response.ok) {
    throw new ExternalServiceError('minicpm-watcher', `MiniCPM watcher answered ${response.status} to a readiness check`, {
      retryable: response.status >= 500,
    });
  }
  const listing = (await response.json().catch(() => null)) as { data?: Array<{ id?: unknown }> } | null;
  const served = (listing?.data ?? []).map((row) => row.id).filter((id): id is string => typeof id === 'string');
  if (!served.includes(config.model)) {
    throw new ExternalServiceError(
      'minicpm-watcher',
      `MiniCPM watcher does not serve "${config.model}" (it serves: ${served.join(', ') || 'nothing'})`,
      { retryable: false },
    );
  }
}
