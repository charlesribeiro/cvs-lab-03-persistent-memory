import { afterEach, describe, expect, it } from 'vitest';
import type Database from 'better-sqlite3';
import { initDb, saveMemory, searchByBm25, searchByVector } from '../src/db/memory-db.js';

describe('memory database', () => {
  let db: Database.Database | undefined;

  afterEach(() => db?.close());

  it('creates the memories table and FTS5 index', () => {
    db = initDb(':memory:');

    const objects = db
      .prepare(`SELECT name, type FROM sqlite_master WHERE name IN ('memories', 'memories_fts') ORDER BY name`)
      .all();

    expect(objects).toEqual([
      { name: 'memories', type: 'table' },
      { name: 'memories_fts', type: 'table' },
    ]);
  });

  it('saves embeddings as JSON and indexes content and category in FTS', () => {
    db = initDb(':memory:');

    const saved = saveMemory(db, {
      userId: 'alice',
      content: 'Prefers dark roast coffee',
      category: 'preference',
      embedding: [1, 0, 0],
    });

    expect(saved).toMatchObject({
      id: expect.any(Number),
      user_id: 'alice',
      content: 'Prefers dark roast coffee',
      category: 'preference',
      embedding: [1, 0, 0],
      access_count: 0,
      last_accessed: null,
    });
    expect(saved.created_at).toEqual(expect.any(String));

    const stored = db.prepare('SELECT embedding FROM memories WHERE id = ?').get(saved.id) as {
      embedding: string;
    };
    expect(stored.embedding).toBe('[1,0,0]');

    const ftsMatch = db
      .prepare(`SELECT rowid, content, category FROM memories_fts WHERE memories_fts MATCH 'coffee'`)
      .get();
    expect(ftsMatch).toEqual({
      rowid: saved.id,
      content: 'Prefers dark roast coffee',
      category: 'preference',
    });
    expect(
      db.prepare(`SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'preference'`).get(),
    ).toEqual({ rowid: saved.id });
  });

  it('keeps the FTS index synchronized on update and delete', () => {
    db = initDb(':memory:');
    const saved = saveMemory(db, {
      userId: 'alice',
      content: 'Likes tea',
      category: 'preference',
      embedding: [1],
    });

    db.prepare('UPDATE memories SET content = ? WHERE id = ?').run('Likes espresso', saved.id);
    expect(db.prepare(`SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'tea'`).all()).toEqual([]);
    expect(db.prepare(`SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'espresso'`).all()).toEqual([
      { rowid: saved.id },
    ]);

    db.prepare('DELETE FROM memories WHERE id = ?').run(saved.id);
    expect(db.prepare(`SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'espresso'`).all()).toEqual([]);
  });


  it('retrieves BM25 matches only for the requested user', () => {
    db = initDb(':memory:');
    saveMemory(db, { userId: 'alice', content: 'Coffee brewing guide', category: 'note', embedding: [1, 0] });
    saveMemory(db, { userId: 'alice', content: 'Tea steeping guide', category: 'note', embedding: [0, 1] });
    saveMemory(db, { userId: 'bob', content: 'Secret coffee order', category: 'private', embedding: [1, 0] });

    const results = searchByBm25(db, 'alice', 'coffee');

    expect(results).toHaveLength(1);
    expect(results[0]).toMatchObject({ content: 'Coffee brewing guide', category: 'note' });
    expect(results[0]?.bm25_score).toBeGreaterThan(0);
    expect(results.some((result) => result.content.includes('Secret'))).toBe(false);
  });

  it('orders vector matches by manually calculated cosine similarity and isolates users', () => {
    db = initDb(':memory:');
    const exact = saveMemory(db, { userId: 'alice', content: 'Exact', category: 'fact', embedding: [1, 0] });
    const angled = saveMemory(db, { userId: 'alice', content: 'Angled', category: 'fact', embedding: [1, 1] });
    saveMemory(db, { userId: 'alice', content: 'Opposite', category: 'fact', embedding: [-1, 0] });
    saveMemory(db, { userId: 'bob', content: 'Bob exact', category: 'fact', embedding: [1, 0] });

    const results = searchByVector(db, 'alice', [1, 0]);

    expect(results.map((result) => result.id)).toEqual([exact.id, angled.id, expect.any(Number)]);
    expect(results.map((result) => result.vector_score)).toEqual([1, expect.closeTo(Math.SQRT1_2, 10), -1]);
    expect(results.every((result) => !result.content.startsWith('Bob'))).toBe(true);
  });

});
