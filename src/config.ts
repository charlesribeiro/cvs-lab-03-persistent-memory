export const DEFAULT_MEMORY_BOOTSTRAP_RELEVANCE_THRESHOLD = 0.08;

export interface AppConfig {
  databasePath: string;
  ollamaBaseUrl: string;
  ollamaEmbeddingModel: string;
  geminiApiKey?: string;
  geminiModel: string;
  memoryBootstrapRelevanceThreshold: number;
  port: number;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  const parsedPort = Number.parseInt(env.PORT ?? '8000', 10);
  if (!Number.isInteger(parsedPort) || parsedPort < 1 || parsedPort > 65_535) {
    throw new Error(`Invalid PORT: ${env.PORT ?? ''}`);
  }
  const bootstrapThreshold = Number(
    env.MEMORY_BOOTSTRAP_RELEVANCE_THRESHOLD ??
      DEFAULT_MEMORY_BOOTSTRAP_RELEVANCE_THRESHOLD,
  );
  if (!Number.isFinite(bootstrapThreshold) || bootstrapThreshold < 0 || bootstrapThreshold > 1) {
    throw new Error(
      `Invalid MEMORY_BOOTSTRAP_RELEVANCE_THRESHOLD: ${env.MEMORY_BOOTSTRAP_RELEVANCE_THRESHOLD ?? ''}`,
    );
  }

  return {
    databasePath: env.DATABASE_PATH ?? './data/memories.db',
    ollamaBaseUrl: env.OLLAMA_BASE_URL ?? 'http://localhost:11434',
    ollamaEmbeddingModel: env.OLLAMA_EMBEDDING_MODEL ?? 'all-minilm',
    ...(env.GOOGLE_API_KEY ? { geminiApiKey: env.GOOGLE_API_KEY } : {}),
    geminiModel: env.GEMINI_MODEL ?? 'gemini-3.1-flash-lite',
    memoryBootstrapRelevanceThreshold: bootstrapThreshold,
    port: parsedPort,
  };
}
