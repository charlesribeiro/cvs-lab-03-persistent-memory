import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { serve } from '@hono/node-server';
import { createGeminiConversationalAgent } from './agent/conversational-agent.js';
import { createApp } from './app.js';
import { loadConfig } from './config.js';
import { initDb } from './db/memory-db.js';
import { OllamaEmbeddings } from './embeddings/embeddings.js';

const config = loadConfig();
if (config.databasePath !== ':memory:') {
  mkdirSync(dirname(config.databasePath), { recursive: true });
}

const db = initDb(config.databasePath);
const embeddings = new OllamaEmbeddings({
  baseUrl: config.ollamaBaseUrl,
  model: config.ollamaEmbeddingModel,
});
const agent = createGeminiConversationalAgent({ db, embeddings, env: process.env });
const app = createApp({ agent });

const server = serve({ fetch: app.fetch, port: config.port }, ({ port }) => {
  console.log(`CVS Lab 03 API listening on http://localhost:${port}`);
});

let shuttingDown = false;
function shutdown(signal: string): void {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`Received ${signal}; shutting down.`);
  server.close(() => {
    db.close();
    process.exit(0);
  });
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
