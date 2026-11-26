import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

process.env.MINICPM_WATCHER_URL = 'https://gpu.example.test:8000/v1';
process.env.MINICPM_WATCHER_API_KEY = 'test-key';
process.env.MINICPM_WATCHER_MODEL = 'openbmb/MiniCPM-V-4.6';

const { buildWindowRequest, readWindowReply, judgeWindow, assertMiniCpmWatcherReady, MINICPM_V46_STOP_TOKEN_IDS } =
  await import('../src/services/minicpm/watcher.js');

const frame = (timestampMs: number, durationMs = 1000) => ({
  timestampMs,
  durationMs,
  encoding: 'jpeg' as const,
  image: Buffer.from(`frame-${timestampMs}`),
});

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('what the watcher is shown', () => {
  it('sends every frame with its time, then the search, and the model\'s own stop tokens', () => {
    const body = buildWindowRequest({ frames: [frame(8000), frame(9000)], query: 'a dog catches a frisbee', model: 'openbmb/MiniCPM-V-4.6' });
    expect(body.model).toBe('openbmb/MiniCPM-V-4.6');
    expect(body.temperature).toBe(0);
    expect(body.stop_token_ids).toEqual([...MINICPM_V46_STOP_TOKEN_IDS]);
    const content = (body.messages as Array<{ role: string; content: unknown }>)[1]!.content as Array<Record<string, unknown>>;
    expect(content[0]).toEqual({ type: 'text', text: 'Frame at 8.0s:' });
    expect((content[1]!.image_url as { url: string }).url).toBe(`data:image/jpeg;base64,${Buffer.from('frame-8000').toString('base64')}`);
    expect(content[2]).toEqual({ type: 'text', text: 'Frame at 9.0s:' });
    const ask = content.at(-1)!.text as string;
    // The search is passed through verbatim, and the window's own range bounds the answer.
    expect(ask).toContain('Search: a dog catches a frisbee');
    expect(ask).toContain('from 8.0s to 10.0s');
    expect(ask).toContain('{"matches": []}');
  });
});

describe('reading what the watcher said', () => {
  const window = { startMs: 8000, endMs: 16000, frameMs: 1000 };

  it('keeps a well-formed match, with how sure it said it was', () => {
    const read = readWindowReply('{"matches":[{"start":9,"end":11.5,"description":" a dog leaps ","confidence":0.8}]}', window);
    expect(read.moments).toEqual([{ startSeconds: 9, endSeconds: 11.5, description: 'a dog leaps', confidence: 0.8 }]);
    expect(read.outsideWindow).toBe(0);
  });

  it('drops a match it could not have seen, and counts it, rather than shifting it into the window', () => {
    const read = readWindowReply('{"matches":[{"start":30,"end":31,"description":"later"},{"start":9,"end":10,"description":"here"}]}', window);
    expect(read.moments.map((m) => m.description)).toEqual(['here']);
    expect(read.outsideWindow).toBe(1);
  });

  it('holds a match that overhangs to the window it was shown', () => {
    const read = readWindowReply('{"matches":[{"start":6,"end":20,"description":"throughout"}]}', window);
    expect(read.moments[0]).toMatchObject({ startSeconds: 8, endSeconds: 16 });
  });

  it('gives a single-instant match the span of a frame, never zero length', () => {
    const read = readWindowReply('{"matches":[{"start":12,"end":12,"description":"the catch"}]}', window);
    expect(read.moments[0]).toMatchObject({ startSeconds: 12, endSeconds: 13 });
  });

  it('carries confidence only when it is usable; zero is a real answer, nonsense is absent', () => {
    const read = readWindowReply(
      '{"matches":[{"start":9,"end":10,"confidence":0},{"start":10,"end":11,"confidence":4.2},{"start":11,"end":12,"confidence":"very"},{"start":12,"end":13}]}',
      window,
    );
    expect(read.moments.map((m) => m.confidence)).toEqual([0, undefined, undefined, undefined]);
    expect('confidence' in read.moments[1]!).toBe(false);
  });

  it('reads an empty list as a window that was looked at and showed nothing', () => {
    expect(readWindowReply('Sure! {"matches": []}', window).moments).toEqual([]);
  });

  it('refuses a reply that is not the expected JSON, so the window is unexamined, not empty', () => {
    expect(() => readWindowReply('I see a dog.', window)).toThrow(/not JSON/);
    expect(() => readWindowReply('{"found": true}', window)).toThrow(/"matches"/);
    expect(() => readWindowReply('{"matches":[{"start":"nine","end":10}]}', window)).toThrow(/malformed/);
  });
});

describe('asking the GPU server', () => {
  it('posts to the chat endpoint with the key, and returns what it read', async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({
      choices: [{ message: { content: '{"matches":[{"start":1,"end":2,"description":"x","confidence":0.5}]}' } }],
      usage: { prompt_tokens: 900, completion_tokens: 30 },
    }), { status: 200 }));
    const judged = await judgeWindow({ frames: [frame(0), frame(1000)], query: 'x' });
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe('https://gpu.example.test:8000/v1/chat/completions');
    expect((init as RequestInit).headers).toMatchObject({ authorization: 'Bearer test-key' });
    expect(judged.moments).toHaveLength(1);
    expect(judged.promptTokens).toBe(900);
  });

  it('marks a server error as worth retrying, and a refusal as not', async () => {
    fetchMock.mockResolvedValueOnce(new Response('overloaded', { status: 503 }));
    await expect(judgeWindow({ frames: [frame(0)], query: 'x' })).rejects.toMatchObject({ retryable: true });
    fetchMock.mockResolvedValueOnce(new Response('bad key', { status: 401 }));
    await expect(judgeWindow({ frames: [frame(0)], query: 'x' })).rejects.toMatchObject({ retryable: false });
  });

  it('treats a reply with no message text as unjudged', async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ choices: [] }), { status: 200 }));
    await expect(judgeWindow({ frames: [frame(0)], query: 'x' })).rejects.toThrow(/no message text/);
  });

  it('is ready only when the server actually serves the model we will ask', async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ data: [{ id: 'openbmb/MiniCPM-V-4.6' }] }), { status: 200 }));
    await expect(assertMiniCpmWatcherReady()).resolves.toBeUndefined();
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ data: [{ id: 'some-other-model' }] }), { status: 200 }));
    await expect(assertMiniCpmWatcherReady()).rejects.toThrow(/does not serve "openbmb\/MiniCPM-V-4.6" \(it serves: some-other-model\)/);
  });
});
