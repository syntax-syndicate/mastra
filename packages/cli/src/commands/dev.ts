import { config } from 'dotenv';
import { execa } from 'execa';
import { writeFileSync } from 'fs';
import { join } from 'path';
import path from 'path';
import { fileURLToPath } from 'url';

import fsExtra from 'fs-extra/esm';
import fs from 'fs/promises';

import { FileService } from '../services/service.file.js';
import { bundle, bundleServer } from '../utils/bundle.js';

import { EXPRESS_SERVER } from './deploy/server.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export async function dev({
  port,
  env,
  dir,
  toolsDir,
}: {
  dir?: string;
  port: number;
  env: Record<string, any>;
  toolsDir?: string;
}) {
  const dotMastraPath = join(process.cwd(), '.mastra');
  const playgroundServePath = join(dotMastraPath, 'playground');
  const key = env[0]?.name;
  const value = env[0]?.value;

  // Copy playground dist files
  await fsExtra.copy(join(path.dirname(path.dirname(__dirname)), 'src/playground/dist'), playgroundServePath, {
    overwrite: true,
  });

  try {
    const fileService = new FileService();
    const envFile = fileService.getFirstExistingFile(['.env.development', '.env']);
    config({ path: envFile });
  } catch (err) {
    //create .env file
    await fsExtra.ensureFile('.env');
    await fs.writeFile(path.join(process.cwd(), '.env'), `${key}=${value}`);
  }

  const dirPath = dir || path.join(process.cwd(), 'src/mastra');
  await bundle(dirPath);

  const defaultToolsPath = path.join(dirPath, 'tools');
  const toolsPath = toolsDir || defaultToolsPath;

  await bundle(toolsPath, {
    outfile: join(dotMastraPath, 'tools.mjs'),
  });

  writeFileSync(join(dotMastraPath, 'index.mjs'), EXPRESS_SERVER);

  await bundleServer(join(dotMastraPath, 'index.mjs'));

  const proc = execa('node', ['server.mjs'], {
    cwd: dotMastraPath,
    env: {
      port: `${port} || 4111`,
      MASTRA_TOOLS: toolsDir ? 'true' : 'false',
      MASTRA_TOOLS_PATH: toolsDir ? join(dotMastraPath, 'tools.mjs') : '',
    },
  });

  proc.stdout.pipe(process.stdout);
  proc.stderr.pipe(process.stderr);

  return proc;
}
