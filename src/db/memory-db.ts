import Database from 'better-sqlite3';
import { dirname } from 'node:path';
import { mkdirSync } from 'node:fs';

export function initDb(databasePath = './data/memories.db'): Database.Database {
  if (databasePath !== ':memory:') {
    mkdirSync(dirname(databasePath), { recursive: true });
  }

  const db = new Database(databasePath);
  db.pragma('foreign_keys = ON');
  db.pragma('journal_mode = WAL');

  db.exec(`
    CREATE TABLE IF NOT EXISTS memories (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT NOT NULL,
      content TEXT NOT NULL,
      category TEXT NOT NULL,
      embedding TEXT NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      access_count INTEGER NOT NULL DEFAULT 0,
      last_accessed TIMESTAMP
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
      content,
      category,
      content='memories',
      content_rowid='id',
      tokenize='porter'
    );

    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
      INSERT INTO memories_fts(rowid, content, category)
      VALUES (new.id, new.content, new.category);
    END;

    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
      INSERT INTO memories_fts(memories_fts, rowid, content, category)
      VALUES ('delete', old.id, old.content, old.category);
    END;

    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
      INSERT INTO memories_fts(memories_fts, rowid, content, category)
      VALUES ('delete', old.id, old.content, old.category);
      INSERT INTO memories_fts(rowid, content, category)
      VALUES (new.id, new.content, new.category);
    END;
  `);

  return db;
}


export interface SaveMemoryInput {
  userId: string;
  content: string;
  category: string;
  embedding: number[];
}

export interface MemoryRecord {
  id: number;
  user_id: string;
  content: string;
  category: string;
  embedding: number[];
  created_at: string;
  access_count: number;
  last_accessed: string | null;
}

interface StoredMemory extends Omit<MemoryRecord, 'embedding'> {
  embedding: string;
}

function deserializeMemory(row: StoredMemory): MemoryRecord {
  return { ...row, embedding: JSON.parse(row.embedding) as number[] };
}

export function saveMemory(db: Database.Database, input: SaveMemoryInput): MemoryRecord {
  const result = db
    .prepare(
      `INSERT INTO memories (user_id, content, category, embedding)
       VALUES (@userId, @content, @category, @embedding)`,
    )
    .run({ ...input, embedding: JSON.stringify(input.embedding) });

  const row = db.prepare('SELECT * FROM memories WHERE id = ?').get(result.lastInsertRowid) as StoredMemory;
  return deserializeMemory(row);
}


export interface SearchResultBase {
  id: number;
  content: string;
  category: string;
  created_at: string;
}

export interface VectorSearchResult extends SearchResultBase {
  vector_score: number;
}

export interface Bm25SearchResult extends SearchResultBase {
  bm25_score: number;
}

export function cosineSimilarity(left: number[], right: number[]): number {
  if (left.length !== right.length) {
    throw new Error(`Embedding dimension mismatch: ${left.length} !== ${right.length}`);
  }
  if (left.length === 0) return 0;

  let dot = 0;
  let leftMagnitude = 0;
  let rightMagnitude = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index] ?? 0;
    const rightValue = right[index] ?? 0;
    dot += leftValue * rightValue;
    leftMagnitude += leftValue * leftValue;
    rightMagnitude += rightValue * rightValue;
  }

  if (leftMagnitude === 0 || rightMagnitude === 0) return 0;
  return dot / (Math.sqrt(leftMagnitude) * Math.sqrt(rightMagnitude));
}

function normalizedLimit(limit: number): number {
  return Math.max(0, Math.floor(limit));
}

export function searchByVector(
  db: Database.Database,
  userId: string,
  queryEmbedding: number[],
  limit = 10,
): VectorSearchResult[] {
  const rows = db
    .prepare(
      `SELECT id, content, category, created_at, embedding
       FROM memories
       WHERE user_id = ?`,
    )
    .all(userId) as Array<SearchResultBase & { embedding: string }>;

  return rows
    .map(({ embedding, ...row }) => ({
      ...row,
      vector_score: cosineSimilarity(queryEmbedding, JSON.parse(embedding) as number[]),
    }))
    .sort((left, right) => right.vector_score - left.vector_score || left.id - right.id)
    .slice(0, normalizedLimit(limit));
}

function toFtsQuery(query: string): string | null {
  const tokens = query.match(/[\p{L}\p{N}_]+/gu);
  return tokens?.length ? tokens.map((token) => `"${token.replaceAll('"', '""')}"`).join(' AND ') : null;
}

export function searchByBm25(
  db: Database.Database,
  userId: string,
  query: string,
  limit = 10,
): Bm25SearchResult[] {
  const ftsQuery = toFtsQuery(query);
  if (!ftsQuery || limit <= 0) return [];

  return db
    .prepare(
      `SELECT m.id, m.content, m.category, m.created_at, -bm25(memories_fts) AS bm25_score
       FROM memories_fts
       JOIN memories AS m ON m.id = memories_fts.rowid
       WHERE memories_fts MATCH ? AND m.user_id = ?
       ORDER BY bm25(memories_fts), m.id
       LIMIT ?`,
    )
    .all(ftsQuery, userId, normalizedLimit(limit)) as Bm25SearchResult[];
}
