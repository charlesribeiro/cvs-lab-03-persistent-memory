export const DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434';
export const DEFAULT_OLLAMA_EMBEDDING_MODEL = 'all-minilm';

export interface EmbeddingProvider {
  getEmbedding(text: string): Promise<number[]>;
}

export interface OllamaEmbeddingOptions {
  baseUrl?: string;
  model?: string;
  fetchImpl?: typeof fetch;
}

interface OllamaEmbedResponse {
  embeddings?: unknown;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export class OllamaEmbeddings implements EmbeddingProvider {
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: OllamaEmbeddingOptions = {}) {
    this.baseUrl = (options.baseUrl ?? DEFAULT_OLLAMA_BASE_URL).replace(/\/+$/, '');
    this.model = options.model ?? DEFAULT_OLLAMA_EMBEDDING_MODEL;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async getEmbedding(text: string): Promise<number[]> {
    if (text.trim().length === 0) {
      throw new Error('Embedding input must not be empty');
    }

    let response: Response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}/api/embed`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: this.model, input: text }),
      });
    } catch (error) {
      throw new Error(`Unable to reach Ollama at ${this.baseUrl}: ${errorMessage(error)}`, {
        cause: error,
      });
    }

    if (!response.ok) {
      const details = (await response.text()).trim();
      throw new Error(
        `Ollama embedding request failed with HTTP ${response.status}${details ? `: ${details}` : ''}`,
      );
    }

    let payload: OllamaEmbedResponse;
    try {
      payload = (await response.json()) as OllamaEmbedResponse;
    } catch (error) {
      throw new Error(`Ollama returned malformed JSON: ${errorMessage(error)}`, { cause: error });
    }

    const embedding = Array.isArray(payload.embeddings) ? payload.embeddings[0] : undefined;
    if (
      !Array.isArray(embedding) ||
      embedding.length === 0 ||
      !embedding.every((value) => typeof value === 'number' && Number.isFinite(value))
    ) {
      throw new Error(
        'Ollama returned a malformed embedding: expected a non-empty array of finite numbers',
      );
    }

    return embedding;
  }
}

let defaultEmbeddingProvider: EmbeddingProvider | undefined;

export function getDefaultEmbeddingProvider(): EmbeddingProvider {
  defaultEmbeddingProvider ??= new OllamaEmbeddings({
    baseUrl: process.env.OLLAMA_BASE_URL ?? DEFAULT_OLLAMA_BASE_URL,
    model: process.env.OLLAMA_EMBEDDING_MODEL ?? DEFAULT_OLLAMA_EMBEDDING_MODEL,
  });
  return defaultEmbeddingProvider;
}
