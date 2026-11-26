import { env } from '../../../config/env.js';
import { ExternalServiceError } from '../../../lib/errors.js';
import { logger } from '../../../lib/logger.js';
import { assertMiniCpmWatcherReady, judgeWindow, type WindowFrame } from '../../minicpm/watcher.js';
import type { VideoModelAdapter, VideoMoment, VideoWatchResult } from '../model.js';
import type { VideoFrame, VideoSourceKind } from '../source.js';
import { isFrameStreamVideoSource } from '../source.js';

const DEFAULT_MAX_EVENTS = 64;

interface Window {
  bucket: number;
  frames: VideoFrame[];
}

interface Range {
  startSeconds: number;
  endSeconds: number;
  reason: string;
}

/** At most `count` frames from a window, spread evenly and always keeping its first and last. */
export function spreadFrames<T>(frames: T[], count: number): T[] {
  if (frames.length <= count) return frames;
  if (count <= 1) return [frames[0]!];
  const picked: T[] = [];
  for (let i = 0; i < count; i += 1) picked.push(frames[Math.round((i * (frames.length - 1)) / (count - 1))]!);
  return picked;
}

function windowRange(window: Window): { startSeconds: number; endSeconds: number } {
  const first = window.frames[0]!;
  const last = window.frames[window.frames.length - 1]!;
  return { startSeconds: first.timestampMs / 1000, endSeconds: (last.timestampMs + last.durationMs) / 1000 };
}

/**
 * MiniCPM-V 4.6 watching a live browser stream for the internet search.
 *
 * The browser plays in real time and hands over timestamped frames. They are
 * grouped into fixed windows of source time (MINICPM_WATCH_WINDOW_SECONDS),
 * and each window is shown to the model as a few evenly spread frames
 * together, with the search. Several scouts each run one of these at once;
 * the vLLM server batches their requests on one GPU, which is how one GPU
 * serves several watchers.
 *
 * Each watcher judges its windows one at a time, in order. If the model falls
 * behind the player by more than MINICPM_WATCH_MAX_PENDING_WINDOWS windows,
 * the oldest waiting window is skipped so the watcher stays near what is
 * playing. Skipped and failed windows are recorded as unexamined, with their
 * time ranges, and make the watch partial — never "watched, nothing there".
 */
export const miniCpmWatcherAdapter: VideoModelAdapter = {
  id: 'minicpm-v-4.6',
  sourceKinds: new Set(['frame-stream']),

  async assertReady(kind: VideoSourceKind): Promise<void> {
    if (kind !== 'frame-stream') {
      throw new ExternalServiceError('minicpm-watcher', `the MiniCPM watcher reads live frame streams, not "${kind}"`, { retryable: false });
    }
    await assertMiniCpmWatcherReady();
  },

  async watch({ source, query, signal, maxEvents, onMoment }): Promise<VideoWatchResult> {
    if (!isFrameStreamVideoSource(source)) {
      throw new ExternalServiceError('minicpm-watcher', `the MiniCPM watcher cannot read source kind "${source.kind}"`, { retryable: false });
    }
    const eventCap = maxEvents ?? DEFAULT_MAX_EVENTS;
    const windowMs = env.MINICPM_WATCH_WINDOW_SECONDS * 1000;
    const framesPerWindow = env.MINICPM_WATCH_FRAMES_PER_WINDOW;
    const maxPending = env.MINICPM_WATCH_MAX_PENDING_WINDOWS;

    const controller = new AbortController();
    const forwardAbort = () => controller.abort(signal?.reason ?? new Error('live watch cancelled'));
    if (signal?.aborted) forwardAbort(); else signal?.addEventListener('abort', forwardAbort, { once: true });

    const moments: VideoMoment[] = [];
    const skipped: Range[] = [];
    const failed: Range[] = [];
    const pending: Window[] = [];
    let current: Window | null = null;
    let producerDone = false;
    let capped = false;
    let framesReceived = 0;
    let framesSent = 0;
    let framesProcessed = 0;
    let framesDropped = 0;
    let windowsJudged = 0;
    let outsideWindow = 0;
    let repairedReplies = 0;
    let latencyTotalMs = 0;
    let latestProducedMs = 0;
    let latestProcessedMs = 0;
    let maxLagMs = 0;
    let watchedThroughSeconds = 0;
    let wake: (() => void) | null = null;
    const nudge = () => { const w = wake; wake = null; w?.(); };

    const enqueue = (window: Window) => {
      pending.push(window);
      while (pending.length > maxPending) {
        const dropped = pending.shift()!;
        framesDropped += dropped.frames.length;
        skipped.push({ ...windowRange(dropped), reason: 'the watcher fell behind the player' });
      }
      nudge();
    };

    // The browser side: frames into windows of source time. Its failure is
    // caught where it starts and kept for the end, so a page that cannot be
    // played does not become an unhandled rejection while the model is busy.
    let browserFailure: unknown = null;
    const producer = (async () => {
      try {
        for await (const frame of source.open(controller.signal)) {
          if (controller.signal.aborted) break;
          framesReceived += 1;
          latestProducedMs = Math.max(latestProducedMs, frame.timestampMs + frame.durationMs);
          maxLagMs = Math.max(maxLagMs, latestProducedMs - latestProcessedMs);
          const bucket = Math.floor(frame.timestampMs / windowMs);
          if (current && bucket !== current.bucket) {
            enqueue(current);
            current = null;
          }
          if (!current) current = { bucket, frames: [] };
          current.frames.push(frame);
        }
      } catch (error) {
        if (!controller.signal.aborted) browserFailure = error;
      } finally {
        if (current && current.frames.length > 0 && !controller.signal.aborted) enqueue(current);
        current = null;
        producerDone = true;
        nudge();
      }
    })();

    // The model side: one window at a time, in order.
    while (true) {
      if (pending.length === 0) {
        if (producerDone || controller.signal.aborted) break;
        await new Promise<void>((resolve) => { wake = resolve; });
        continue;
      }
      const window = pending.shift()!;
      const range = windowRange(window);
      const frames = spreadFrames(window.frames, framesPerWindow);
      try {
        const judged = await judgeWindow({
          frames: frames.map((frame): WindowFrame => ({ timestampMs: frame.timestampMs, durationMs: frame.durationMs, encoding: frame.encoding, image: frame.image })),
          query,
          signal: controller.signal,
        });
        windowsJudged += 1;
        framesSent += frames.length;
        framesProcessed += window.frames.length;
        outsideWindow += judged.outsideWindow;
        if (judged.repaired) repairedReplies += 1;
        latencyTotalMs += judged.latencyMs;
        latestProcessedMs = Math.max(latestProcessedMs, range.endSeconds * 1000);
        watchedThroughSeconds = Math.max(watchedThroughSeconds, range.endSeconds);
        for (const moment of judged.moments) {
          if (moments.length >= eventCap) break;
          moments.push(moment);
          await onMoment?.(moment);
        }
        if (moments.length >= eventCap) {
          capped = true;
          controller.abort(new Error('the watcher reached its moment ceiling'));
          break;
        }
      } catch (error) {
        if (controller.signal.aborted && !capped) break;
        const reason = error instanceof Error ? error.message : String(error);
        failed.push({ ...range, reason });
        framesDropped += window.frames.length;
        logger.warn('MiniCPM window could not be judged', { window_start_s: range.startSeconds, window_end_s: range.endSeconds, reason });
      }
    }

    controller.abort(new Error('live watch closed'));
    signal?.removeEventListener('abort', forwardAbort);
    await producer;
    // Whatever was still waiting when the watch stopped was never looked at.
    for (const window of pending.splice(0)) {
      framesDropped += window.frames.length;
      skipped.push({ ...windowRange(window), reason: capped ? 'the watcher reached its moment ceiling' : 'the watch stopped first' });
    }

    // A search that was called off ends the watch with its reason, as the other watcher does.
    if (signal?.aborted) throw signal.reason ?? new Error('live watch cancelled');
    if (browserFailure !== null && windowsJudged === 0) throw browserFailure;
    if (windowsJudged === 0 && failed.length > 0) {
      throw new ExternalServiceError('minicpm-watcher', `no window could be judged: ${failed[0]!.reason}`, { retryable: true });
    }

    const completion = await source.completion.catch((error: unknown) => ({
      exhausted: false,
      reason: error instanceof Error ? error.message : String(error),
      watchedThroughSeconds: latestProducedMs / 1000,
    }));
    const durationSeconds = Math.max(completion.watchedThroughSeconds, latestProducedMs / 1000);
    const exhausted = completion.exhausted && !capped && skipped.length === 0 && failed.length === 0 && browserFailure === null;

    return {
      model: env.MINICPM_WATCHER_MODEL,
      revision: env.MINICPM_WATCHER_REVISION ?? 'unrecorded',
      durationSeconds,
      watchedThroughSeconds,
      exhausted,
      moments,
      metrics: {
        watcher: 'minicpm',
        window_seconds: env.MINICPM_WATCH_WINDOW_SECONDS,
        frames_per_window: framesPerWindow,
        windows_judged: windowsJudged,
        windows_skipped: skipped.length,
        windows_failed: failed.length,
        unexamined: [...skipped, ...failed],
        matches_outside_window: outsideWindow,
        replies_repaired: repairedReplies,
        mean_window_latency_ms: windowsJudged > 0 ? Math.round(latencyTotalMs / windowsJudged) : null,
        frames_received: framesReceived,
        client_frames_sent: framesSent,
        client_frames_processed: framesProcessed,
        client_frames_dropped_for_lag: framesDropped,
        latest_source_ms: latestProducedMs,
        latest_processed_ms: latestProcessedMs,
        max_video_lag_ms: maxLagMs,
        source_reason: completion.reason,
        browser_failure: browserFailure === null ? null : (browserFailure instanceof Error ? browserFailure.message : String(browserFailure)),
        capped,
      },
    };
  },
};
