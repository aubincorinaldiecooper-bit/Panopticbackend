import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { FrameStreamCompletion, FrameStreamVideoSource, VideoFrame } from '../src/services/video/source.js';

process.env.MINICPM_WATCHER_URL = 'https://gpu.example.test:8000/v1';
process.env.MINICPM_WATCHER_API_KEY = 'test-key';
process.env.MINICPM_WATCH_WINDOW_SECONDS = '4';
process.env.MINICPM_WATCH_FRAMES_PER_WINDOW = '2';
process.env.MINICPM_WATCH_MAX_PENDING_WINDOWS = '1';

const judgeWindow = vi.fn();
vi.mock('../src/services/minicpm/watcher.js', () => ({
  judgeWindow: (input: unknown) => judgeWindow(input),
  assertMiniCpmWatcherReady: async () => undefined,
}));
vi.mock('../src/lib/logger.js', () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn(), child: () => ({ info: vi.fn(), warn: vi.fn(), error: vi.fn() }) },
}));

const { miniCpmWatcherAdapter, spreadFrames } = await import('../src/services/video/adapters/minicpm.js');
const { internetWatcher } = await import('../src/services/video/adapters/internetWatcher.js');

const frameAt = (seconds: number): VideoFrame => ({ timestampMs: seconds * 1000, durationMs: 1000, encoding: 'jpeg', image: Buffer.from(String(seconds)) });

/** A browser stream of one frame per second, `count` seconds long, optionally failing after some frames. */
function stream(count: number, options: { exhausted?: boolean; failAfter?: number; between?: () => Promise<void> } = {}): FrameStreamVideoSource {
  let resolveCompletion!: (value: FrameStreamCompletion) => void;
  const completion = new Promise<FrameStreamCompletion>((resolve) => { resolveCompletion = resolve; });
  return {
    kind: 'frame-stream',
    id: 'test',
    completion,
    // Like the real browser source, it always settles how far it got, even when stopped early.
    async *open(signal: AbortSignal) {
      let through = 0;
      let finished = false;
      try {
        for (let s = 0; s < count; s += 1) {
          if (signal.aborted) break;
          if (options.failAfter !== undefined && s >= options.failAfter) throw new Error('the page has no video');
          yield frameAt(s);
          through = s + 1;
          await (options.between?.() ?? Promise.resolve());
        }
        finished = !signal.aborted;
      } finally {
        resolveCompletion(finished
          ? { exhausted: options.exhausted ?? true, reason: 'ended', watchedThroughSeconds: count }
          : { exhausted: false, reason: 'cancelled', watchedThroughSeconds: through });
      }
    },
  };
}

const window = (frames: Array<{ timestampMs: number }>) => frames.map((f) => f.timestampMs / 1000);

beforeEach(() => {
  judgeWindow.mockReset();
});

describe('the MiniCPM watcher on a live browser stream', () => {
  it('is the live search\'s watcher unless VideoChat3 is chosen', () => {
    expect(internetWatcher().id).toBe('minicpm-v-4.6');
  });

  it('spreads a window\'s frames evenly, always keeping its first and last', () => {
    expect(spreadFrames([0, 1, 2, 3, 4, 5, 6, 7], 3)).toEqual([0, 4, 7]);
    expect(spreadFrames([0, 1], 8)).toEqual([0, 1]);
  });

  it('looks at each window of source time in order, and reports moments as it finds them', async () => {
    judgeWindow.mockImplementation(async ({ frames }: { frames: Array<{ timestampMs: number }> }) => ({
      moments: frames[0]!.timestampMs === 4000 ? [{ startSeconds: 5, endSeconds: 6, description: 'found it', confidence: 0.9 }] : [],
      outsideWindow: 0, repaired: false, latencyMs: 10,
    }));
    const seen: string[] = [];
    const result = await miniCpmWatcherAdapter.watch({
      source: stream(8), query: 'q', onMoment: (moment) => { seen.push(moment.description); },
    });
    expect(judgeWindow.mock.calls.map(([input]) => window((input as { frames: Array<{ timestampMs: number }> }).frames))).toEqual([[0, 3], [4, 7]]);
    expect(seen).toEqual(['found it']);
    expect(result.exhausted).toBe(true);
    expect(result.watchedThroughSeconds).toBe(8);
    expect(result.metrics).toMatchObject({ windows_judged: 2, windows_failed: 0, windows_skipped: 0, client_frames_processed: 8, client_frames_sent: 4 });
  });

  it('records a window it could not judge as unexamined, keeps going, and calls the watch partial', async () => {
    judgeWindow
      .mockRejectedValueOnce(new Error('MiniCPM watcher answered 503'))
      .mockResolvedValue({ moments: [], outsideWindow: 0, repaired: false, latencyMs: 10 });
    const result = await miniCpmWatcherAdapter.watch({ source: stream(8), query: 'q' });
    expect(result.exhausted).toBe(false);
    expect(result.metrics.windows_judged).toBe(1);
    expect(result.metrics.unexamined).toEqual([{ startSeconds: 0, endSeconds: 4, reason: 'MiniCPM watcher answered 503' }]);
  });

  it('fails the watch outright when no window could be judged, rather than claiming it watched', async () => {
    judgeWindow.mockRejectedValue(new Error('MiniCPM watcher unreachable: connect ECONNREFUSED'));
    await expect(miniCpmWatcherAdapter.watch({ source: stream(8), query: 'q' })).rejects.toThrow(/no window could be judged: MiniCPM watcher unreachable/);
  });

  it('skips the oldest waiting window when it falls behind the player, and says which', async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    let calls = 0;
    judgeWindow.mockImplementation(async () => {
      calls += 1;
      if (calls === 1) await gate; // the first look is slow while the browser keeps playing
      return { moments: [], outsideWindow: 0, repaired: false, latencyMs: 10 };
    });
    let produced = 0;
    const source = stream(16, {
      between: async () => {
        produced += 1;
        if (produced === 15) release();
      },
    });
    const result = await miniCpmWatcherAdapter.watch({ source, query: 'q' });
    expect(result.exhausted).toBe(false);
    expect(result.metrics.windows_skipped).toBeGreaterThan(0);
    expect((result.metrics.unexamined as Array<{ reason: string }>).every((range) => range.reason === 'the watcher fell behind the player')).toBe(true);
    expect(result.metrics.client_frames_dropped_for_lag).toBe((result.metrics.windows_skipped as number) * 4);
  });

  it('stops at its moment ceiling and does not call the stretch after it watched', async () => {
    judgeWindow.mockResolvedValue({ moments: [{ startSeconds: 1, endSeconds: 2, description: 'x' }], outsideWindow: 0, repaired: false, latencyMs: 10 });
    const result = await miniCpmWatcherAdapter.watch({ source: stream(12), query: 'q', maxEvents: 1 });
    expect(result.moments).toHaveLength(1);
    expect(result.exhausted).toBe(false);
    expect(result.metrics.capped).toBe(true);
  });

  it('is partial when the browser did not complete its stretch, however cleanly each window went', async () => {
    judgeWindow.mockResolvedValue({ moments: [], outsideWindow: 0, repaired: false, latencyMs: 10 });
    const result = await miniCpmWatcherAdapter.watch({ source: stream(8, { exhausted: false }), query: 'q' });
    expect(result.exhausted).toBe(false);
  });

  it('passes on a page that cannot be played before anything was judged', async () => {
    judgeWindow.mockResolvedValue({ moments: [], outsideWindow: 0, repaired: false, latencyMs: 10 });
    await expect(miniCpmWatcherAdapter.watch({ source: stream(8, { failAfter: 0 }), query: 'q' })).rejects.toThrow(/no video/);
  });

  it('ends with the search\'s own reason when the search is called off', async () => {
    const controller = new AbortController();
    judgeWindow.mockImplementation(async () => {
      controller.abort(new Error('search cancelled'));
      throw new Error('aborted');
    });
    await expect(miniCpmWatcherAdapter.watch({ source: stream(8), query: 'q', signal: controller.signal })).rejects.toThrow(/search cancelled/);
  });
});
