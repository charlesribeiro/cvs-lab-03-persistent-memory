import { randomUUID } from 'node:crypto';
import { Hono, type Context } from 'hono';
import { SessionOwnershipError } from './agent/conversational-agent.js';

export interface ChatAgent {
  chat(userId: string, message: string, sessionId: string): Promise<string>;
  resetSession(sessionId: string, userId: string): boolean;
}

export interface CreateAppOptions {
  agent: ChatAgent;
  logger?: (message: string, details?: Record<string, unknown>) => void;
}

type ErrorStatus = 400 | 404 | 409 | 500;

class ApiError extends Error {
  constructor(
    readonly status: ErrorStatus,
    readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

function errorResponse(context: Context, error: ApiError) {
  return context.json(
    { error: { code: error.code, message: error.message } },
    error.status,
  );
}

async function readJsonObject(context: Context): Promise<Record<string, unknown>> {
  const text = await context.req.text();
  if (!text.trim()) {
    throw new ApiError(400, 'missing_body', 'Request body is required.');
  }

  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    throw new ApiError(400, 'invalid_json', 'Request body must be valid JSON.');
  }
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    throw new ApiError(400, 'invalid_request', 'Request body must be a JSON object.');
  }
  return body as Record<string, unknown>;
}

function requiredString(body: Record<string, unknown>, field: string): string {
  const value = body[field];
  if (typeof value !== 'string' || !value.trim()) {
    throw new ApiError(400, 'invalid_request', `${field} is required and must not be blank`);
  }
  return value.trim();
}

function optionalSessionId(body: Record<string, unknown>): string {
  const value = body.session_id;
  if (value === undefined) return randomUUID();
  if (typeof value !== 'string' || !value.trim()) {
    throw new ApiError(400, 'invalid_request', 'session_id must not be blank');
  }
  return value.trim();
}

export function createApp(options: CreateAppOptions): Hono {
  const app = new Hono();
  const logger = options.logger ?? console.error;

  app.get('/health', (context) => context.json({ status: 'ok' }));

  app.post('/chat', async (context) => {
    const body = await readJsonObject(context);
    const userId = requiredString(body, 'user_id');
    const message = requiredString(body, 'message');
    const sessionId = optionalSessionId(body);
    const response = await options.agent.chat(userId, message, sessionId);
    return context.json({ response, session_id: sessionId });
  });

  app.post('/chat/reset', async (context) => {
    const body = await readJsonObject(context);
    const userId = requiredString(body, 'user_id');
    const sessionId = requiredString(body, 'session_id');
    const reset = options.agent.resetSession(sessionId, userId);
    if (!reset) {
      throw new ApiError(404, 'session_not_found', 'Session was not found.');
    }
    return context.json({ status: 'ok', session_id: sessionId });
  });

  app.onError((error, context) => {
    if (error instanceof ApiError) return errorResponse(context, error);
    if (error instanceof SessionOwnershipError) {
      return errorResponse(
        context,
        new ApiError(409, 'session_ownership_conflict', 'Session belongs to a different user.'),
      );
    }
    logger('Unhandled API error', {
      name: error instanceof Error ? error.name : undefined,
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    });
    return errorResponse(
      context,
      new ApiError(500, 'internal_error', 'An unexpected internal error occurred.'),
    );
  });

  return app;
}
