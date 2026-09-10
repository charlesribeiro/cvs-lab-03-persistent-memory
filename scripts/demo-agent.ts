import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createGeminiConversationalAgent } from '../src/agent/conversational-agent.js';
import { loadConfig } from '../src/config.js';
import { initDb } from '../src/db/memory-db.js';
import { getDefaultEmbeddingProvider } from '../src/embeddings/embeddings.js';
import { searchMemoryHybrid } from '../src/memory/memory-tools.js';

const config = loadConfig();
if (!config.geminiApiKey) {
  console.error('GOOGLE_API_KEY is required. Export it before running npm run demo:agent.');
  process.exitCode = 1;
} else {
  const directory = mkdtempSync(join(tmpdir(), 'cvs-lab-03-agent-'));
  const db = initDb(join(directory, 'demo.db'));
  const embeddings = getDefaultEmbeddingProvider();
  const logs: string[] = [];
  const logger = (message: string) => {
    logs.push(message);
    console.error(message);
  };
  const agent = createGeminiConversationalAgent({ db, embeddings, debug: true, logger });

  async function turn(userId: string, sessionId: string, message: string) {
    console.log(`\n[${userId}/${sessionId}] User: ${message}`);
    const answer = await agent.chat(userId, message, sessionId);
    console.log(`[${userId}/${sessionId}] Gemini: ${answer}`);
    return answer;
  }

  try {
    await turn(
      'user1',
      's1',
      'Hi! I prefer Python over TypeScript, and I work at Acme Corp as a senior engineer.',
    );
    await turn(
      'user1',
      's1',
      'I am migrating our monolith to microservices. We use PostgreSQL and Redis.',
    );

    const recallLogStart = logs.length;
    const recommendation = await turn(
      'user1',
      's2',
      'Can you recommend a good framework for my project?',
    );
    const bootstrapLog = logs
      .slice(recallLogStart)
      .find((line) => line.startsWith('[agent] proactive memory bootstrap:'));

    const user2Answer = await turn(
      'user2',
      'user2-session',
      'What do you know about my programming preferences?',
    );
    const user2Results = await searchMemoryHybrid(
      db,
      { userId: 'user2', query: 'programming preferences', limit: 10 },
      embeddings,
    );

    const resetSucceeded = agent.resetSession('s1', 'user1');
    const resetRecall = await turn(
      'user1',
      's1',
      'What programming language do I prefer?',
    );

    const recommendationLower = recommendation.toLowerCase();
    const s2RecalledContext =
      recommendationLower.includes('python') || recommendationLower.includes('fastapi');
    const user1Terms = ['python', 'typescript', 'acme', 'postgresql', 'redis', 'microservice'];
    const user2Lower = user2Answer.toLowerCase();
    const leakedTerms = user1Terms.filter((term) => user2Lower.includes(term));
    const user2IsolationPassed = user2Results.length === 0 && leakedTerms.length === 0;
    const resetRecallPassed = resetSucceeded && resetRecall.toLowerCase().includes('python');

    console.log('\nVerification:');
    console.log(`- New-session proactive bootstrap: ${bootstrapLog ?? 'not observed'}`);
    console.log(`- user1/s2 recalled context: ${s2RecalledContext}`);
    console.log(`- Tailored recommendation: ${recommendation}`);
    console.log(`- User2-scoped database results: ${user2Results.length}`);
    console.log(`- User2 leaked user1 terms: ${leakedTerms.join(', ') || 'none'}`);
    console.log(`- User2 isolation passed: ${user2IsolationPassed}`);
    console.log(`- Reset user1/s1 succeeded: ${resetSucceeded}`);
    console.log(`- Reset + persistent recall passed: ${resetRecallPassed}`);
    console.log(`- Reset recall answer: ${resetRecall}`);

    if (!bootstrapLog || bootstrapLog.endsWith(': 0 relevant')) {
      throw new Error('New user1 session did not bootstrap relevant persistent memory.');
    }
    if (!s2RecalledContext) {
      throw new Error('New user1 session recommendation did not reflect recalled context.');
    }
    if (!user2IsolationPassed) {
      throw new Error(`User isolation check failed; leaked terms: ${leakedTerms.join(', ')}`);
    }
    if (!resetRecallPassed) {
      throw new Error('Reset session did not recover the Python preference from persistent memory.');
    }
  } finally {
    db.close();
    rmSync(directory, { recursive: true, force: true });
  }
}
