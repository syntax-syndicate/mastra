import { createLogger, Mastra, UpstashRedisLogger } from '@mastra/core';

import {agents } from '@/agents';
import * as tools from '@/tools';
import { integrations } from '@/integrations';


export const mastra = new Mastra<typeof integrations, typeof tools, any, UpstashRedisLogger >({
  tools,
  agents,
  logger: createLogger({
    type: 'UPSTASH',
    token: process.env.UPSTASH_API_KEY!,
    url: process.env.UPSTASH_URL!,
    key: 'logs',
  })
});




