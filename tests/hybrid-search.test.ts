import { afterEach, describe, expect, it } from 'vitest';
import type Database from 'better-sqlite3';
import { initDb, saveMemory } from '../src/db/memory-db.js';
import { hybridSearch } from '../src/memory/hybrid-search.js';

describe('hybridSearch', () => {
  let db: Database.Database | undefined;

  afterEach(() => db?.close());

  it('min-max normalizes each source and combines vector 0.7 with BM25 0.3', () => {
    db = initDb(':memory:');
    const vectorOnly = saveMemory(db, {
      userId: 'alice',
      content: 'Prefers espresso',
      category: 'preference',
      embedding: [1, 0],
    });
    const bm25Only = saveMemory(db, {
      userId: 'alice',
      content: 'Coffee brewing notes',
      category: 'note',
      embedding: [0, 1],
    });

    const results = hybridSearch(db, 'alice', 'coffee', [1, 0]);

    expect(results).toHaveLength(2);
    expect(results[0]).toMatchObject({
      id: vectorOnly.id,
      vector_score: 1,
      bm25_score: 0,
      hybrid_score: 0.7,
    });
    expect(results[1]).toMatchObject({
      id: bm25Only.id,
      vector_score: 0,
      bm25_score: 1,
      hybrid_score: 0.3,
    });
  });

  it('never merges in another user’s vector or FTS result', () => {
    db = initDb(':memory:');
    const alice = saveMemory(db, {
      userId: 'alice',
      content: 'Alice coffee preference',
      category: 'preference',
      embedding: [1, 0],
    });
    saveMemory(db, {
      userId: 'bob',
      content: 'Bob secret coffee preference',
      category: 'private',
      embedding: [1, 0],
    });

    const results = hybridSearch(db, 'alice', 'coffee', [1, 0]);

    expect(results.map((result) => result.id)).toEqual([alice.id]);
    expect(results[0]).toMatchObject({ vector_score: 1, bm25_score: 1, hybrid_score: 1 });
  });
});
