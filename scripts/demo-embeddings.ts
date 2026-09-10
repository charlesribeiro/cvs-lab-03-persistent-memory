import { cosineSimilarity, initDb } from '../src/db/memory-db.js';
import {
  DEFAULT_OLLAMA_BASE_URL,
  DEFAULT_OLLAMA_EMBEDDING_MODEL,
  OllamaEmbeddings,
} from '../src/embeddings/embeddings.js';
import {
  saveMemoryWithEmbedding,
  searchMemoryHybrid,
} from '../src/memory/memory-tools.js';

const baseUrl = process.env.OLLAMA_BASE_URL ?? DEFAULT_OLLAMA_BASE_URL;
const model = process.env.OLLAMA_EMBEDDING_MODEL ?? DEFAULT_OLLAMA_EMBEDDING_MODEL;
const embeddings = new OllamaEmbeddings({ baseUrl, model });

const dogText = 'I like dogs';
const puppyText = 'I love puppies';
const marketText = 'The stock market crashed';

const [dogEmbedding, puppyEmbedding, marketEmbedding] = await Promise.all([
  embeddings.getEmbedding(dogText),
  embeddings.getEmbedding(puppyText),
  embeddings.getEmbedding(marketText),
]);
const dogPuppySimilarity = cosineSimilarity(dogEmbedding, puppyEmbedding);
const dogMarketSimilarity = cosineSimilarity(dogEmbedding, marketEmbedding);

console.log(`Ollama: ${baseUrl}`);
console.log(`Model: ${model}`);
console.log(`Vector dimensions: ${dogEmbedding.length}`);
console.log(`similarity("${dogText}", "${puppyText}") = ${dogPuppySimilarity.toFixed(6)}`);
console.log(`similarity("${dogText}", "${marketText}") = ${dogMarketSimilarity.toFixed(6)}`);

if (dogPuppySimilarity <= dogMarketSimilarity) {
  throw new Error('Semantic check failed: dog/puppy similarity was not greater than dog/market');
}

const db = initDb(':memory:');
try {
  const userId = 'embedding-demo-user';
  const memories = [
    ['User prefers Python for backend development', 'preference'],
    ['User works at Acme Corp', 'employment'],
    ['User is migrating a monolith to microservices', 'project'],
    ['User uses PostgreSQL and Redis', 'technology'],
  ] as const;

  for (const [content, category] of memories) {
    await saveMemoryWithEmbedding(db, { userId, content, category }, embeddings);
  }

  const query = 'What programming language does the user like?';
  const results = await searchMemoryHybrid(db, { userId, query, limit: memories.length }, embeddings);
  const pythonRank = results.findIndex((result) => result.content.includes('Python')) + 1;

  console.log(`
Semantic memory query: "${query}"`);
  for (const [index, result] of results.entries()) {
    console.log(
      `${index + 1}. ${result.content} ` +
        `(vector=${result.vector_score.toFixed(6)}, bm25=${result.bm25_score.toFixed(6)}, hybrid=${result.hybrid_score.toFixed(6)})`,
    );
  }
  console.log(`Python memory rank: ${pythonRank}`);

  if (pythonRank < 1 || pythonRank > 2) {
    throw new Error(`Semantic retrieval check failed: Python memory ranked ${pythonRank || 'outside results'}`);
  }
} finally {
  db.close();
}
