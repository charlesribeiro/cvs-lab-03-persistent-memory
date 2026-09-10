import { describe, expect, it, vi } from 'vitest';
import { createApp } from '../src/app.js';
import {
  ConversationalAgent,
  SessionOwnershipError,
  type AgentModel,
} from '../src/agent/conversational-agent.js';
import { initDb, saveMemory, searchByVector } from '../src/db/memory-db.js';

function fakeAgent() {
  return {
    chat: vi.fn().mockResolvedValue('Hello'),
    resetSession: vi.fn().mockReturnValue(true),
  };
}

describe('GET /health', () => {
  it('returns an ok health response', async () => {
    const response = await createApp({ agent: fakeAgent() }).request('/health');

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ status: 'ok' });
  });
});

describe('POST /chat', () => {
  it('returns the agent response with a supplied session ID', async () => {
    const agent = fakeAgent();
    agent.chat.mockResolvedValue('Use FastAPI.');
    const response = await createApp({ agent }).request('/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ user_id: 'user1', message: 'Recommend a framework', session_id: 's1' }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ response: 'Use FastAPI.', session_id: 's1' });
    expect(agent.chat).toHaveBeenCalledWith('user1', 'Recommend a framework', 's1');
  });

  it('generates a UUID session ID and accepts it on the next turn', async () => {
    const agent = fakeAgent();
    agent.chat.mockResolvedValueOnce('First').mockResolvedValueOnce('Second');
    const app = createApp({ agent });
    const first = await app.request('/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ user_id: 'user1', message: 'First turn' }),
    });
    const firstBody = await first.json() as { response: string; session_id: string };

    expect(first.status).toBe(200);
    expect(firstBody.response).toBe('First');
    expect(firstBody.session_id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );

    const second = await app.request('/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        user_id: 'user1',
        message: 'Second turn',
        session_id: firstBody.session_id,
      }),
    });

    expect(second.status).toBe(200);
    await expect(second.json()).resolves.toEqual({
      response: 'Second',
      session_id: firstBody.session_id,
    });
    expect(agent.chat).toHaveBeenNthCalledWith(1, 'user1', 'First turn', firstBody.session_id);
    expect(agent.chat).toHaveBeenNthCalledWith(2, 'user1', 'Second turn', firstBody.session_id);
  });

  it.each([
    [{ message: 'Hello' }, 'user_id is required and must not be blank'],
    [{ user_id: '', message: 'Hello' }, 'user_id is required and must not be blank'],
    [{ user_id: '   ', message: 'Hello' }, 'user_id is required and must not be blank'],
    [{ user_id: 'user1' }, 'message is required and must not be blank'],
    [{ user_id: 'user1', message: '' }, 'message is required and must not be blank'],
    [{ user_id: 'user1', message: '   ' }, 'message is required and must not be blank'],
    [{ user_id: 'user1', message: 'Hello', session_id: ' ' }, 'session_id must not be blank'],
  ])('rejects invalid chat input %#', async (body, message) => {
    const agent = fakeAgent();
    const response = await createApp({ agent }).request('/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: { code: 'invalid_request', message },
    });
    expect(agent.chat).not.toHaveBeenCalled();
  });

  it('rejects malformed JSON safely', async () => {
    const agent = fakeAgent();
    const response = await createApp({ agent }).request('/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: '{not-json',
    });

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: { code: 'invalid_json', message: 'Request body must be valid JSON.' },
    });
    expect(agent.chat).not.toHaveBeenCalled();
  });

  it('rejects a missing request body safely', async () => {
    const agent = fakeAgent();
    const response = await createApp({ agent }).request('/chat', { method: 'POST' });

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: { code: 'missing_body', message: 'Request body is required.' },
    });
    expect(agent.chat).not.toHaveBeenCalled();
  });


  it('rejects cross-user session reuse', async () => {
    const agent = fakeAgent();
    agent.chat.mockRejectedValue(new SessionOwnershipError('s1'));
    const response = await createApp({ agent, logger: vi.fn() }).request('/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ user_id: 'user2', message: 'Hello', session_id: 's1' }),
    });

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual({
      error: {
        code: 'session_ownership_conflict',
        message: 'Session belongs to a different user.',
      },
    });
  });

  it('returns a safe 500 when the agent fails', async () => {
    const agent = fakeAgent();
    const logger = vi.fn();
    agent.chat.mockRejectedValue(new Error('GOOGLE_API_KEY=your_google_ai_api_key'));
    const response = await createApp({ agent, logger }).request('/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ user_id: 'user1', message: 'Hello', session_id: 's1' }),
    });

    expect(response.status).toBe(500);
    const body = await response.json();
    expect(body).toEqual({
      error: { code: 'internal_error', message: 'An unexpected internal error occurred.' },
    });
    expect(JSON.stringify(body)).not.toContain('secret-value');
    expect(logger).toHaveBeenCalledOnce();
  });
});


describe('POST /chat/reset', () => {
  it('resets an owned session', async () => {
    const agent = fakeAgent();
    const response = await createApp({ agent }).request('/chat/reset', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ user_id: 'user1', session_id: 's1' }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ status: 'ok', session_id: 's1' });
    expect(agent.resetSession).toHaveBeenCalledWith('s1', 'user1');
  });

  it('returns 404 for an unknown session', async () => {
    const agent = fakeAgent();
    agent.resetSession.mockReturnValue(false);
    const response = await createApp({ agent }).request('/chat/reset', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ user_id: 'user1', session_id: 'missing' }),
    });

    expect(response.status).toBe(404);
    await expect(response.json()).resolves.toEqual({
      error: { code: 'session_not_found', message: 'Session was not found.' },
    });
  });
});


describe('POST /chat/reset validation and safety', () => {
  it.each([
    [{ session_id: 's1' }, 'user_id is required and must not be blank'],
    [{ user_id: '', session_id: 's1' }, 'user_id is required and must not be blank'],
    [{ user_id: '   ', session_id: 's1' }, 'user_id is required and must not be blank'],
    [{ user_id: 'user1' }, 'session_id is required and must not be blank'],
    [{ user_id: 'user1', session_id: '' }, 'session_id is required and must not be blank'],
    [{ user_id: 'user1', session_id: '   ' }, 'session_id is required and must not be blank'],
  ])('rejects invalid reset input %#', async (body, message) => {
    const agent = fakeAgent();
    const response = await createApp({ agent }).request('/chat/reset', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: { code: 'invalid_request', message },
    });
    expect(agent.resetSession).not.toHaveBeenCalled();
  });

  it('rejects malformed reset JSON', async () => {
    const agent = fakeAgent();
    const response = await createApp({ agent }).request('/chat/reset', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: '{bad-json',
    });

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: { code: 'invalid_json', message: 'Request body must be valid JSON.' },
    });
  });

  it('rejects a missing reset body', async () => {
    const response = await createApp({ agent: fakeAgent() }).request('/chat/reset', {
      method: 'POST',
    });

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: { code: 'missing_body', message: 'Request body is required.' },
    });
  });

  it.each([null, [], 'reset'])('rejects non-object reset JSON: %j', async (body) => {
    const response = await createApp({ agent: fakeAgent() }).request('/chat/reset', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });

    expect(response.status).toBe(400);
    await expect(response.json()).resolves.toEqual({
      error: { code: 'invalid_request', message: 'Request body must be a JSON object.' },
    });
  });

  it('rejects a cross-user reset with 409', async () => {
    const agent = fakeAgent();
    agent.resetSession.mockImplementation(() => {
      throw new SessionOwnershipError('s1');
    });
    const response = await createApp({ agent, logger: vi.fn() }).request('/chat/reset', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ user_id: 'user2', session_id: 's1' }),
    });

    expect(response.status).toBe(409);
    await expect(response.json()).resolves.toEqual({
      error: {
        code: 'session_ownership_conflict',
        message: 'Session belongs to a different user.',
      },
    });
  });

  it('returns a safe 500 when reset fails unexpectedly', async () => {
    const agent = fakeAgent();
    const logger = vi.fn();
    agent.resetSession.mockImplementation(() => {
      throw new Error('GOOGLE_API_KEY=secret-value');
    });
    const response = await createApp({ agent, logger }).request('/chat/reset', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ user_id: 'user1', session_id: 's1' }),
    });

    expect(response.status).toBe(500);
    const body = await response.json();
    expect(body).toEqual({
      error: { code: 'internal_error', message: 'An unexpected internal error occurred.' },
    });
    expect(JSON.stringify(body)).not.toContain('secret-value');
    expect(logger).toHaveBeenCalledWith('Unhandled API error', {
      name: 'Error',
      message: 'GOOGLE_API_KEY=secret-value',
      stack: expect.any(String),
    });
  });
});

describe('reset persistence', () => {
  it('clears session state without deleting SQLite memory and bootstraps after recreation', async () => {
    const db = initDb(':memory:');
    try {
      saveMemory(db, {
        userId: 'user1',
        content: 'User prefers Python over TypeScript.',
        category: 'preference',
        embedding: [1, 0],
      });
      const chats: Array<Array<string | unknown[]>> = [];
      const model: AgentModel = {
        startChat() {
          const requests: Array<string | unknown[]> = [];
          chats.push(requests);
          return {
            async sendMessage(request) {
              requests.push(request);
              return {
                response: {
                  text: () => 'You prefer Python.',
                  functionCalls: () => undefined,
                },
              };
            },
          };
        },
      };
      const agent = new ConversationalAgent({
        model,
        db,
        embeddings: { getEmbedding: vi.fn().mockResolvedValue([1, 0]) },
      });
      const app = createApp({ agent });

      const first = await app.request('/chat', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ user_id: 'user1', message: 'What language?', session_id: 's1' }),
      });
      expect(first.status).toBe(200);
      expect(chats).toHaveLength(1);

      const reset = await app.request('/chat/reset', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ user_id: 'user1', session_id: 's1' }),
      });
      expect(reset.status).toBe(200);
      expect(searchByVector(db, 'user1', [1, 0], 10)).toHaveLength(1);

      const recreated = await app.request('/chat', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          user_id: 'user1',
          message: 'What programming language do I prefer?',
          session_id: 's1',
        }),
      });

      expect(recreated.status).toBe(200);
      await expect(recreated.json()).resolves.toEqual({
        response: 'You prefer Python.',
        session_id: 's1',
      });
      expect(chats).toHaveLength(2);
      expect(String(chats[1]?.[0])).toContain('User prefers Python over TypeScript.');
      expect(searchByVector(db, 'user1', [1, 0], 10)).toHaveLength(1);
    } finally {
      db.close();
    }
  });
});
