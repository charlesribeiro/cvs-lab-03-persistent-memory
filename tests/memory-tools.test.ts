import { afterEach, describe, expect, it, vi } from 'vitest';
import type Database from 'better-sqlite3';
import { initDb, saveMemory } from '../src/db/memory-db.js';
import type { EmbeddingProvider } from '../src/embeddings/embeddings.js';
import {
  saveMemoryWithEmbedding,
  searchMemoryHybrid,
} from '../src/memory/memory-tools.js';

describe('embedding-backed memory helpers', () => {
  let db: Database.Database | undefined;

  afterEach(() => db?.close());

  it('embeds content before delegating persistence to saveMemory', async () => {
    db = initDb(':memory:');
    const getEmbedding = vi.fn().mockResolvedValue([0.25, 0.75]);
    const provider: EmbeddingProvider = { getEmbedding };

    const saved = await saveMemoryWithEmbedding(
      db,
      { userId: 'alice', content: 'Uses TypeScript', category: 'preference' },
      provider,
    );

    expect(getEmbedding).toHaveBeenCalledWith('Uses TypeScript');
    expect(saved).toMatchObject({
      user_id: 'alice',
      content: 'Uses TypeScript',
      category: 'preference',
      embedding: [0.25, 0.75],
    });
  });

  it('embeds the query before delegating to deterministic hybrid search', async () => {
    db = initDb(':memory:');
    const preferred = saveMemory(db, {
      userId: 'alice',
      content: 'Prefers Python for backend work',
      category: 'preference',
      embedding: [1, 0],
    });
    saveMemory(db, {
      userId: 'alice',
      content: 'Works at Acme Corp',
      category: 'employment',
      embedding: [0, 1],
    });
    const getEmbedding = vi.fn().mockResolvedValue([1, 0]);
    const provider: EmbeddingProvider = { getEmbedding };

    const results = await searchMemoryHybrid(
      db,
      { userId: 'alice', query: 'Which language does the user like?', limit: 2 },
      provider,
    );

    expect(getEmbedding).toHaveBeenCalledWith('Which language does the user like?');
    expect(results[0]?.id).toBe(preferred.id);
    expect(results[0]?.hybrid_score).toBe(0.7);
  });
});
