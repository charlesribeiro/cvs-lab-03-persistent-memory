import { describe, expect, it, vi } from 'vitest';
import { OllamaEmbeddings } from '../src/embeddings/embeddings.js';

function response(body: unknown, status = 200): Response {
  return new Response(typeof body === 'string' ? body : JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

describe('OllamaEmbeddings', () => {
  it('uses the modern Ollama endpoint and returns its first embedding', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      response({ embeddings: [[0.1, -0.2, 0.3]] }),
    );
    const provider = new OllamaEmbeddings({ fetchImpl: fetchMock });

    await expect(provider.getEmbedding('I like dogs')).resolves.toEqual([0.1, -0.2, 0.3]);
    expect(fetchMock).toHaveBeenCalledWith('http://localhost:11434/api/embed', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'all-minilm', input: 'I like dogs' }),
    });
  });

  it.each([
    {},
    { embeddings: [] },
    { embeddings: [[]] },
    { embeddings: [[0.1, 'bad']] },
    { embeddings: [[0.1, Number.NaN]] },
  ])('rejects malformed embedding response %#', async (payload) => {
    const provider = new OllamaEmbeddings({
      fetchImpl: vi.fn<typeof fetch>().mockResolvedValue(response(payload)),
    });

    await expect(provider.getEmbedding('valid text')).rejects.toThrow(
      'Ollama returned a malformed embedding: expected a non-empty array of finite numbers',
    );
  });

  it('reports HTTP failures with status and response details', async () => {
    const provider = new OllamaEmbeddings({
      fetchImpl: vi.fn<typeof fetch>().mockResolvedValue(response('model not found', 404)),
    });

    await expect(provider.getEmbedding('valid text')).rejects.toThrow(
      'Ollama embedding request failed with HTTP 404: model not found',
    );
  });

  it('reports when Ollama is unavailable', async () => {
    const provider = new OllamaEmbeddings({
      fetchImpl: vi.fn<typeof fetch>().mockRejectedValue(new TypeError('fetch failed')),
    });

    await expect(provider.getEmbedding('valid text')).rejects.toThrow(
      'Unable to reach Ollama at http://localhost:11434: fetch failed',
    );
  });

  it.each(['', '   ', '\n\t'])('rejects empty input before making a request', async (text) => {
    const fetchMock = vi.fn<typeof fetch>();
    const provider = new OllamaEmbeddings({ fetchImpl: fetchMock });

    await expect(provider.getEmbedding(text)).rejects.toThrow('Embedding input must not be empty');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('honors configured base URL and model', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(response({ embeddings: [[1]] }));
    const provider = new OllamaEmbeddings({
      baseUrl: 'http://ollama.internal:9999/',
      model: 'custom-model',
      fetchImpl: fetchMock,
    });

    await provider.getEmbedding('hello');

    expect(fetchMock).toHaveBeenCalledWith(
      'http://ollama.internal:9999/api/embed',
      expect.objectContaining({ body: JSON.stringify({ model: 'custom-model', input: 'hello' }) }),
    );
  });
});
