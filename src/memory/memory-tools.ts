import type Database from 'better-sqlite3';
import {
  SchemaType,
  type FunctionDeclaration,
} from '@google/generative-ai';
import { saveMemory } from '../db/memory-db.js';
import {
  getDefaultEmbeddingProvider,
  type EmbeddingProvider,
} from '../embeddings/embeddings.js';
import { hybridSearch } from './hybrid-search.js';

export const MEMORY_CATEGORIES = [
  'preference',
  'fact',
  'decision',
  'context',
  'general',
] as const;

export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

export const MEMORY_FUNCTION_DECLARATIONS: FunctionDeclaration[] = [
  {
    name: 'memory_search',
    description:
      'Use this when the user references previous conversations, preferences, known facts, decisions, projects, or when previous context would help answer correctly.',
    parameters: {
      type: SchemaType.OBJECT,
      properties: {
        query: {
          type: SchemaType.STRING,
          description: 'A focused semantic query describing the prior user context needed.',
        },
        limit: {
          type: SchemaType.INTEGER,
          description: 'Optional maximum number of memories to return.',
        },
      },
      required: ['query'],
    },
  },
  {
    name: 'memory_save',
    description:
      'Use this when the user shares information that is important and likely useful in future conversations. Avoid greetings, transient statements, trivial information, and redundant memories already known.',
    parameters: {
      type: SchemaType.OBJECT,
      properties: {
        content: {
          type: SchemaType.STRING,
          description: 'A concise, self-contained durable memory about the user.',
        },
        category: {
          type: SchemaType.STRING,
          format: 'enum',
          enum: [...MEMORY_CATEGORIES],
          description: 'The durable-memory category.',
        },
      },
      required: ['content', 'category'],
    },
  },
];

export interface SaveMemoryWithEmbeddingInput {
  userId: string;
  content: string;
  category: string;
}

export interface SearchMemoryHybridInput {
  userId: string;
  query: string;
  limit?: number;
}

export interface MemoryToolDependencies {
  db: Database.Database;
  embeddings?: EmbeddingProvider;
}

export async function saveMemoryWithEmbedding(
  db: Database.Database,
  input: SaveMemoryWithEmbeddingInput,
  embeddings: EmbeddingProvider = getDefaultEmbeddingProvider(),
) {
  const embedding = await embeddings.getEmbedding(input.content);
  return saveMemory(db, { ...input, embedding });
}

export async function searchMemoryHybrid(
  db: Database.Database,
  input: SearchMemoryHybridInput,
  embeddings: EmbeddingProvider = getDefaultEmbeddingProvider(),
) {
  const queryEmbedding = await embeddings.getEmbedding(input.query);
  return hybridSearch(db, input.userId, input.query, queryEmbedding, input.limit ?? 10);
}

function record(input: unknown): Record<string, unknown> {
  return typeof input === 'object' && input !== null && !Array.isArray(input)
    ? (input as Record<string, unknown>)
    : {};
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function searchLimit(value: unknown): number | null {
  if (value === undefined) return 10;
  return typeof value === 'number' && Number.isInteger(value) && value > 0
    ? Math.min(value, 50)
    : null;
}

function formattedTimestamp(value: string): string {
  const isoCandidate = value.includes('T') ? value : `${value.replace(' ', 'T')}Z`;
  const date = new Date(isoCandidate);
  return Number.isNaN(date.valueOf()) ? value : date.toISOString();
}

export async function executeMemoryTool(
  name: string,
  input: unknown,
  userId: string,
  dependencies: MemoryToolDependencies,
): Promise<string> {
  const args = record(input);
  const embeddings = dependencies.embeddings ?? getDefaultEmbeddingProvider();

  if (name === 'memory_search') {
    const query = nonEmptyString(args.query);
    if (!query) return 'Error: memory_search requires a non-empty query.';
    const limit = searchLimit(args.limit);
    if (limit === null) return 'Error: memory_search limit must be a positive integer.';

    const results = await searchMemoryHybrid(
      dependencies.db,
      { userId, query, limit },
      embeddings,
    );
    if (results.length === 0) return 'No relevant memories found.';

    return results
      .map(
        (result) =>
          `[Score: ${result.hybrid_score.toFixed(2)}] (${result.category})\n` +
          `${result.content}\nSaved: ${formattedTimestamp(result.created_at)}`,
      )
      .join('\n\n');
  }

  if (name === 'memory_save') {
    const content = nonEmptyString(args.content);
    if (!content) return 'Error: memory_save requires non-empty content.';
    const category = nonEmptyString(args.category);
    if (!category || !MEMORY_CATEGORIES.includes(category as MemoryCategory)) {
      return `Error: invalid memory category. Allowed categories: ${MEMORY_CATEGORIES.join(', ')}.`;
    }

    const saved = await saveMemoryWithEmbedding(
      dependencies.db,
      { userId, content, category },
      embeddings,
    );
    return `Memory saved (id: ${saved.id}, category: ${saved.category}).`;
  }

  return `Error: unknown memory tool "${name}".`;
}

export function createMemoryTools(
  db: Database.Database,
  embeddings: EmbeddingProvider = getDefaultEmbeddingProvider(),
) {
  return {
    save(userId: string, content: string, category: string) {
      return saveMemoryWithEmbedding(db, { userId, content, category }, embeddings);
    },
    search(userId: string, query: string, limit = 10) {
      return searchMemoryHybrid(db, { userId, query, limit }, embeddings);
    },
    execute(name: string, input: unknown, userId: string) {
      return executeMemoryTool(name, input, userId, { db, embeddings });
    },
  };
}
