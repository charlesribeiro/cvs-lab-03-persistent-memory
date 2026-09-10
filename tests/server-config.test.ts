import { describe, expect, it } from 'vitest';
import { loadConfig } from '../src/config.js';

describe('server configuration', () => {
  it('uses port 8000 by default', () => {
    expect(loadConfig({}).port).toBe(8000);
  });

  it('accepts a valid PORT override', () => {
    expect(loadConfig({ PORT: '8123' }).port).toBe(8123);
  });
});
