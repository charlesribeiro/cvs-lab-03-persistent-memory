import { resolve } from 'node:path';
import { initDb, saveMemory } from '../src/db/memory-db.js';
import { hybridSearch } from '../src/memory/hybrid-search.js';

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

const databasePath = argument('--db') ?? process.env.DATABASE_PATH ?? './data/memories.db';
const db = initDb(databasePath);

try {
  if (process.argv.includes('--seed')) {
    saveMemory(db, {
      userId: 'demo-user',
      content: 'Prefers dark roast coffee',
      category: 'preference',
      embedding: [1, 0, 0],
    });
    saveMemory(db, {
      userId: 'demo-user',
      content: 'Keeps brewing notes',
      category: 'note',
      embedding: [0.7, 0.7, 0],
    });
  }

  const tables = db
    .prepare(`SELECT name, type FROM sqlite_master WHERE name IN ('memories', 'memories_fts') ORDER BY name`)
    .all();
  const triggers = db
    .prepare(`SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'memories' ORDER BY name`)
    .all();
  const memoryCount = db.prepare('SELECT COUNT(*) AS count FROM memories').get();
  const ftsCount = db.prepare('SELECT COUNT(*) AS count FROM memories_fts').get();
  const recent = db
    .prepare(`SELECT id, user_id, content, category, embedding, created_at, access_count, last_accessed
              FROM memories ORDER BY id DESC LIMIT 10`)
    .all();

  console.log(JSON.stringify({
    database: databasePath === ':memory:' ? databasePath : resolve(databasePath),
    tables,
    triggers,
    memoryCount,
    ftsCount,
    recent,
    ...(process.argv.includes('--seed')
      ? { demoHybridSearch: hybridSearch(db, 'demo-user', 'coffee', [1, 0, 0]) }
      : {}),
  }, null, 2));
} finally {
  db.close();
}
