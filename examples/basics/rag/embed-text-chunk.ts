import { MDocument, embed } from '@mastra/rag';

const doc = MDocument.fromText('Your text content...');

const chunks = await doc.chunk();

const { embedding } = await embed(chunks[0], {
  provider: 'OPEN_AI',
  model: 'text-embedding-ada-002',
  maxRetries: 3,
});

console.log(embedding);