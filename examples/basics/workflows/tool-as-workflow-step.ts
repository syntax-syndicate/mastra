import { createTool, Workflow } from '@mastra/core';
import { z } from 'zod';

const crawlWebpage = createTool({
  id: 'Crawl Webpage',
  description: 'Crawls a webpage and extracts the text content',
  inputSchema: z.object({
    url: z.string().url(),
  }),
  outputSchema: z.object({
    rawText: z.string(),
  }),
  execute: async ({ context: { url } }) => {
    return { rawText: 'This is the text content of the webpage' };
  },
});

const contentWorkflow = new Workflow({ name: 'content-review' });

contentWorkflow.step(crawlWebpage);
contentWorkflow.commit();
const res = await contentWorkflow.execute();

console.log(res.results);
