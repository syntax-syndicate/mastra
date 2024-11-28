import { CoreMessage, UserContent } from 'ai';
import { Integration } from '../integration';
import { createLogger, Logger, RegisteredLogger } from '../logger';
import { AllTools, ToolApi } from '../tools/types';
import { LLM } from '../llm';
import { ModelConfig, StructuredOutput } from '../llm/types';

export class Agent<
  TTools,
  TIntegrations extends Integration[] | undefined = undefined,
  TKeys extends keyof AllTools<TTools, TIntegrations> = keyof AllTools<
    TTools,
    TIntegrations
  >,
> {
  public name: string;
  readonly llm: LLM<TTools, TIntegrations, TKeys>;
  readonly instructions: string;
  readonly model: ModelConfig;
  readonly enabledTools: Partial<Record<TKeys, boolean>>;
  logger: Logger;
  logGroupId?: string;
  constructor(config: {
    name: string;
    instructions: string;
    model: ModelConfig;
    enabledTools?: Partial<Record<TKeys, boolean>>;
    logGroupId?: string;
  }) {
    this.name = config.name;
    this.instructions = config.instructions;

    this.llm = new LLM<TTools, TIntegrations, TKeys>();

    this.model = config.model;
    this.enabledTools = config.enabledTools || {};
    this.logger = createLogger({ type: 'CONSOLE' });
    this.logger.info(
      `Agent ${this.name} initialized with model ${this.model.provider}`
    );
  }

  /**
   * Set the concrete tools for the agent
   * @param tools
   */
  __setTools(tools: Record<TKeys, ToolApi>) {
    this.llm.__setTools(tools);
    this.log(`Tools set for agent ${this.name}`)
  }

  /**
   * Set the logger for the agent
   * @param logger
   */
  __setLogger(logger: Logger) {
    this.logger = logger;
    this.log(`Logger updated for agent ${this.name}`)
  }

  private log(message: string) {
    this.logger.info({
      message,
      destinationPath: this.name,
      type: RegisteredLogger.AGENT,
      logGroupId: this.logGroupId,
    });
  }

  // private error(message: string) {
  //   this.logger.error({
  //     message,
  //     destinationPath: this.name,
  //     type: RegisteredLogger.AGENT
  //   });
  // }

  async text({
    messages,
    onStepFinish,
    maxSteps = 5,
  }: {
    messages: UserContent[];
    onStepFinish?: (step: string) => void;
    maxSteps?: number;
  }) {
    this.log(`Starting text generation for agent ${this.name}`);

    const systemMessage: CoreMessage = {
      role: 'system',
      content: this.instructions,
    };

    const userMessages: CoreMessage[] = messages.map((content) => ({
      role: 'user',
      content: content,
    }));

    const messageObjects = [systemMessage, ...userMessages];

    return this.llm.text({
      model: this.model,
      messages: messageObjects,
      enabledTools: this.enabledTools,
      onStepFinish,
      maxSteps,
    });
  }

  async textObject({
    messages,
    structuredOutput,
    onStepFinish,
    maxSteps = 5,
  }: {
    messages: UserContent[];
    structuredOutput: StructuredOutput;
    onStepFinish?: (step: string) => void;
    maxSteps?: number;
  }) {
    this.log(`Starting text generation for agent ${this.name}`);

    const systemMessage: CoreMessage = {
      role: 'system',
      content: this.instructions,
    };

    const userMessages: CoreMessage[] = messages.map((content) => ({
      role: 'user',
      content: content,
    }));

    const messageObjects = [systemMessage, ...userMessages];

    return this.llm.textObject({
      model: this.model,
      messages: messageObjects,
      structuredOutput,
      enabledTools: this.enabledTools,
      onStepFinish,
      maxSteps,
    });
  }

  async stream({
    messages,
    onStepFinish,
    onFinish,
    maxSteps = 5,
  }: {
    messages: UserContent[];
    onStepFinish?: (step: string) => void;
    onFinish?: (result: string) => Promise<void> | void;
    maxSteps?: number;
  }) {
    this.log(`Starting stream generation for agent ${this.name}`);

    const systemMessage: CoreMessage = {
      role: 'system',
      content: this.instructions,
    };

    const userMessages: CoreMessage[] = messages.map((content) => ({
      role: 'user',
      content: content,
    }));

    const messageObjects = [systemMessage, ...userMessages];

    return this.llm.stream({
      messages: messageObjects,
      model: this.model,
      enabledTools: this.enabledTools,
      onStepFinish,
      onFinish,
      maxSteps,
    });
  }

  async streamObject({
    messages,
    structuredOutput,
    onStepFinish,
    onFinish,
    maxSteps = 5,
  }: {
    messages: UserContent[];
    structuredOutput: StructuredOutput;
    onStepFinish?: (step: string) => void;
    onFinish?: (result: string) => Promise<void> | void;
    maxSteps?: number;
  }) {
    this.log(`Starting stream generation for agent ${this.name}`);

    const systemMessage: CoreMessage = {
      role: 'system',
      content: this.instructions,
    };

    const userMessages: CoreMessage[] = messages.map((content) => ({
      role: 'user',
      content: content,
    }));

    const messageObjects = [systemMessage, ...userMessages];

    return this.llm.streamObject({
      messages: messageObjects,
      structuredOutput,
      model: this.model,
      enabledTools: this.enabledTools,
      onStepFinish,
      onFinish,
      maxSteps,
    });
  }
}
