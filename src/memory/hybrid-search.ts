import type Database from 'better-sqlite3';
import {
  searchByBm25,
  searchByVector,
  type SearchResultBase,
} from '../db/memory-db.js';

export const VECTOR_WEIGHT = 0.7;
export const BM25_WEIGHT = 0.3;

export interface HybridSearchResult extends SearchResultBase {
  /** Raw cosine similarity, retained for absolute relevance filtering. */
  semantic_score: number;
  /** Min-max-normalized vector rank used by the hybrid scorer. */
  vector_score: number;
  bm25_score: number;
  hybrid_score: number;
}

function minMaxById(items: Array<{ id: number; score: number }>): Map<number, number> {
  if (items.length === 0) return new Map();

  const values = items.map((item) => item.score);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const range = maximum - minimum;

  return new Map(items.map((item) => [item.id, range === 0 ? 1 : (item.score - minimum) / range]));
}

export function hybridSearch(
  db: Database.Database,
  userId: string,
  query: string,
  queryEmbedding: number[],
  limit = 10,
): HybridSearchResult[] {
  if (limit <= 0) return [];

  const vectorResults = searchByVector(db, userId, queryEmbedding, limit);
  const bm25Results = searchByBm25(db, userId, query, limit);
  const semanticScores = new Map(
    vectorResults.map((result) => [result.id, result.vector_score]),
  );
  const vectorScores = minMaxById(
    vectorResults.map((result) => ({ id: result.id, score: result.vector_score })),
  );
  const bm25Scores = minMaxById(
    bm25Results.map((result) => ({ id: result.id, score: result.bm25_score })),
  );

  const memories = new Map<number, SearchResultBase>();
  for (const { vector_score: _score, ...memory } of vectorResults) memories.set(memory.id, memory);
  for (const { bm25_score: _score, ...memory } of bm25Results) memories.set(memory.id, memory);

  return [...memories.values()]
    .map((memory) => {
      const vector_score = vectorScores.get(memory.id) ?? 0;
      const bm25_score = bm25Scores.get(memory.id) ?? 0;
      return {
        ...memory,
        semantic_score: semanticScores.get(memory.id) ?? 0,
        vector_score,
        bm25_score,
        hybrid_score: VECTOR_WEIGHT * vector_score + BM25_WEIGHT * bm25_score,
      };
    })
    .sort((left, right) => right.hybrid_score - left.hybrid_score || left.id - right.id)
    .slice(0, Math.floor(limit));
}
