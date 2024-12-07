import { Integration } from '../integration';
import { Agent } from '../agent';
import { BaseLogger, createLogger } from '../logger';
import { AllTools, ToolApi } from '../tools/types';
import { MastraEngine } from '../engine';
import { MastraVector } from '../vector';
import { LLM } from '../llm';
import { z } from 'zod';
import { syncApi } from '../sync/types';
import { StripUndefined } from './types';
import { Run } from '../run/types';

export class Mastra<
  TIntegrations extends Integration[],
  MastraTools extends Record<string, any>,
  TSyncs extends Record<string, syncApi<any, any>>,
  TLogger extends BaseLogger = BaseLogger,
> {
  private engine?: MastraEngine;
  private vectors?: Record<string, MastraVector>;
  private tools: AllTools<MastraTools, TIntegrations>;
  private agents: Map<string, Agent<MastraTools, TIntegrations>>;
  llm: LLM<
    MastraTools,
    TIntegrations,
    keyof AllTools<MastraTools, TIntegrations>
  >;
  private integrations: Map<string, Integration>;
  private logger: TLogger;
  private syncs: TSyncs;

  constructor(config: {
    tools?: MastraTools;
    syncs?: TSyncs;
    agents?: Agent<MastraTools, TIntegrations>[];
    integrations?: TIntegrations;
    engine?: MastraEngine;
    vectors?: Record<string, MastraVector>;
    logger?: TLogger;
  }) {
    /* 
    Logger
    */

    let logger = createLogger({ type: 'CONSOLE' }) as TLogger;

    if (config.logger) {
      logger = config.logger;
    }

    this.logger = logger;

    /* 
    Integrations
    */

    this.integrations = new Map();

    config.integrations?.forEach((integration) => {
      if (this.integrations.has(integration.name)) {
        throw new Error(
          `Integration with name ${integration.name} already exists`
        );
      }
      this.integrations.set(integration.name, integration);
    });

    const integrationTools =
      config.integrations?.reduce(
        (acc, integration) => ({
          ...acc,
          ...integration.tools,
        }),
        {}
      ) || {};

    const configuredTools = config?.tools || {};

    // Merge custom tools with integration tools
    const allTools = {
      ...configuredTools,
      ...integrationTools,
    } as MastraTools;

    // Hydrate tools with integration tools
    const hydratedTools = Object.entries(allTools ?? {}).reduce<
      Record<string, ToolApi>
    >((memo, [key, val]) => {
      memo[key] = {
        ...val,
        executor: (params) => {
          return val.executor({
            ...params,
            integrationsRegistry: () => ({
              get: <I extends TIntegrations[number]['name']>(name: I) =>
                this.getIntegration(name) as Extract<
                  TIntegrations[number],
                  { name: I }
                >,
            }),
            agents: this.agents,
            llm: this.llm,
            engine: this.engine,
            vectors: this.vectors,
          });
        },
      };
      return memo;
    }, {});

    this.tools = hydratedTools as AllTools<MastraTools, TIntegrations>;

    /* 
    LLM
    */

    this.llm = new LLM<
      MastraTools,
      TIntegrations,
      keyof AllTools<MastraTools, TIntegrations>
    >();
    this.llm.__setTools(this.tools);
    const llmLogger = this.getLogger();
    if (llmLogger) {
      this.llm.__setLogger(llmLogger);
    }

    /* 
    Agents
    */

    this.agents = new Map();

    config.agents?.forEach((agent) => {
      if (this.agents.has(agent.name)) {
        throw new Error(`Agent with name ${agent.name} already exists`);
      }
      this.agents.set(agent.name, agent);
      agent.__setTools(this.tools);
      const agentLogger = this.getLogger();
      if (agentLogger) {
        agent.__setLogger(agentLogger);
      }
    });

    /* 
    Syncs
    */

    if (config.syncs && !config.engine) {
      throw new Error('Engine is required to run syncs');
    }

    this.syncs = (config.syncs || {}) as TSyncs;

    /* 
    Engine
    */

    if (config.engine) {
      this.engine = config.engine;
    }

    /* 
    Vectors
    */

    if (config.vectors) {
      this.vectors = config.vectors;
    }
  }

  public async sync<K extends keyof TSyncs>(
    key: K,
    params: TSyncs[K]['schema']['_input'],
    runId?: Run['runId']
  ): Promise<StripUndefined<TSyncs[K]['outputShema']>['_input']> {
    if (!this.engine) {
      throw new Error(`Engine is required to run syncs`);
    }

    const sync = this.syncs?.[key];

    if (!sync) {
      throw new Error(`Sync function ${key as string} not found`);
    }

    const syncFn = sync['executor'];

    if (!syncFn) {
      throw new Error(`Sync function ${key as string} not found`);
    }

    return await syncFn({
      data: params,
      runId,
      engine: this.engine,
      agents: this.agents,
      vectors: this.vectors,
      llm: this.llm,
      integrationsRegistry: () => ({
        get: <I extends TIntegrations[number]['name']>(name: I) =>
          this.getIntegration(name) as Extract<
            TIntegrations[number],
            { name: I }
          >,
      }),
      toolsRegistry: <T>() => ({
        get: <N extends keyof T>(name: N) =>
          this.getTool(name as string) as T[N],
      }),
    });
  }

  public getAgent(name: string) {
    const agent = this.agents.get(name);
    if (!agent) {
      throw new Error(`Agent with name ${name} not found`);
    }
    return agent;
  }

  public getIntegration<I extends TIntegrations[number]['name']>(name: I) {
    const stringifiedName = String(name);
    const integration = this.integrations.get(stringifiedName.toUpperCase());

    if (!integration) {
      throw new Error(`Integration with name ${stringifiedName} not found`);
    }
    return integration as Extract<TIntegrations[number], { name: I }>;
  }

  public getLLM() {
    return this.llm;
  }

  public getTool<T extends keyof MastraTools>(name: T) {
    const tools = this.tools as MastraTools;
    const tool = tools[name];

    if (!tool) {
      throw new Error(`Tool with name ${String(name)} not found`);
    }

    return {
      ...tool,
      execute: async <
        IN extends MastraTools[T]['schema'],
        OUT extends StripUndefined<MastraTools[T]['outputSchema']>,
      >(
        params: z.infer<IN>,
        runId?: Run['runId']
      ): Promise<z.infer<OUT>> => {
        return tool.executor({
          data: params,
          runId,
          integrationsRegistry: () => ({
            get: <I extends TIntegrations[number]['name']>(name: I) =>
              this.getIntegration(name) as Extract<
                TIntegrations[number],
                { name: I }
              >,
          }),
          agents: this.agents,
          llm: this.llm,
          engine: this.engine,
        });
      },
    };
  }

  public availableIntegrations() {
    return Array.from(this.integrations.entries()).map(
      ([name, integration]) => {
        return {
          name,
          integration,
        };
      }
    );
  }

  public getTools() {
    return this.tools;
  }

  public setLogger({ logger }: { logger: TLogger }) {
    this.logger = logger;
  }

  public getLogger() {
    return this.logger;
  }

  public async getLogsByRunId(runId: string) {
    return await this.logger.getLogsByRunId(runId);
  }
}
