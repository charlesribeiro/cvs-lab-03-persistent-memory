import { afterEach, describe, expect, it, vi } from 'vitest';
import type Database from 'better-sqlite3';
import { SchemaType } from '@google/generative-ai';
import { initDb, saveMemory } from '../src/db/memory-db.js';
import type { EmbeddingProvider } from '../src/embeddings/embeddings.js';
import {
  MEMORY_FUNCTION_DECLARATIONS,
  executeMemoryTool,
} from '../src/memory/memory-tools.js';

describe('Gemini memory function declarations', () => {
  it('declares memory_search and memory_save with required schemas', () => {
    expect(MEMORY_FUNCTION_DECLARATIONS).toHaveLength(2);
    const search = MEMORY_FUNCTION_DECLARATIONS.find((tool) => tool.name === 'memory_search');
    const save = MEMORY_FUNCTION_DECLARATIONS.find((tool) => tool.name === 'memory_save');

    expect(search?.parameters).toMatchObject({
      type: SchemaType.OBJECT,
      required: ['query'],
      properties: {
        query: { type: SchemaType.STRING },
        limit: { type: SchemaType.INTEGER },
      },
    });
    expect(save?.parameters).toMatchObject({
      type: SchemaType.OBJECT,
      required: ['content', 'category'],
      properties: {
        content: { type: SchemaType.STRING },
        category: {
          type: SchemaType.STRING,
          format: 'enum',
          enum: ['preference', 'fact', 'decision', 'context', 'general'],
        },
      },
    });
  });
});

describe('executeMemoryTool', () => {
  let db: Database.Database | undefined;
  const embeddings: EmbeddingProvider = { getEmbedding: vi.fn().mockResolvedValue([1, 0]) };

  afterEach(() => {
    db?.close();
    vi.clearAllMocks();
  });

  it('dispatches memory_search and formats only the requested user memories', async () => {
    db = initDb(':memory:');
    saveMemory(db, {
      userId: 'alice',
      content: 'User prefers Python for backend development',
      category: 'preference',
      embedding: [1, 0],
    });
    saveMemory(db, {
      userId: 'bob',
      content: 'User prefers Rust',
      category: 'preference',
      embedding: [1, 0],
    });

    const output = await executeMemoryTool(
      'memory_search',
      { query: 'backend language', limit: 5 },
      'alice',
      { db, embeddings },
    );

    expect(embeddings.getEmbedding).toHaveBeenCalledWith('backend language');
    expect(output).toContain('[Score: 0.70] (preference)');
    expect(output).toContain('User prefers Python for backend development');
    expect(output).toContain('Saved:');
    expect(output).not.toContain('Rust');
  });

  it('dispatches memory_save with the caller userId', async () => {
    db = initDb(':memory:');

    const output = await executeMemoryTool(
      'memory_save',
      { content: 'User works at Acme Corp', category: 'fact' },
      'alice',
      { db, embeddings },
    );

    expect(output).toMatch(/Memory saved \(id: \d+, category: fact\)\./);
    const row = db.prepare('SELECT user_id, content, category FROM memories').get();
    expect(row).toEqual({
      user_id: 'alice',
      content: 'User works at Acme Corp',
      category: 'fact',
    });
  });

  it('rejects invalid memory categories without embedding or saving', async () => {
    db = initDb(':memory:');

    const output = await executeMemoryTool(
      'memory_save',
      { content: 'Secret', category: 'employment' },
      'alice',
      { db, embeddings },
    );

    expect(output).toContain('Error: invalid memory category');
    expect(embeddings.getEmbedding).not.toHaveBeenCalled();
    expect(db.prepare('SELECT count(*) AS count FROM memories').get()).toEqual({ count: 0 });
  });

  it('returns safe errors for invalid search and unknown tools', async () => {
    db = initDb(':memory:');

    await expect(
      executeMemoryTool('memory_search', { query: '   ' }, 'alice', { db, embeddings }),
    ).resolves.toBe('Error: memory_search requires a non-empty query.');
    await expect(
      executeMemoryTool('delete_everything', {}, 'alice', { db, embeddings }),
    ).resolves.toBe('Error: unknown memory tool "delete_everything".');
  });
});
