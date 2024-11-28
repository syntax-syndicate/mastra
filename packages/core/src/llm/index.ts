import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAI } from '@ai-sdk/openai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { createMistral } from '@ai-sdk/mistral';
import { createAmazonBedrock } from '@ai-sdk/amazon-bedrock';
import { createAzure } from '@ai-sdk/azure';
import { createXai } from '@ai-sdk/xai';
import { createCohere } from '@ai-sdk/cohere';
import { createAnthropicVertex } from 'anthropic-vertex-ai';
import {
  CoreMessage,
  CoreTool as CT,
  embed,
  embedMany,
  EmbeddingModel,
  generateObject,
  generateText,
  LanguageModelV1,
  streamObject,
  streamText,
  tool,
} from 'ai';
import { z, ZodSchema } from 'zod';
import { AllTools, CoreTool, ToolApi } from '../tools/types';
import { delay } from '../utils';
import { Integration } from '../integration';
import { createLogger, Logger, LogLevel, RegisteredLogger, BaseLogMessage } from '../logger';
import {
  CustomModelConfig,
  EmbeddingModelConfig,
  GoogleGenerativeAISettings,
  LLMProvider,
  ModelConfig,
  StructuredOutput,
  StructuredOutputType,
} from './types';

export class LLM<
  TTools,
  TIntegrations extends Integration[] | undefined = undefined,
  TKeys extends keyof AllTools<TTools, TIntegrations> = keyof AllTools<
    TTools,
    TIntegrations
  >,
> {
  #tools: Record<TKeys, ToolApi>;
  #logger: Logger;
  logGroupId?: string;
  constructor({logGroupId}: {logGroupId?: string} = {}) {
    this.#tools = {} as Record<TKeys, ToolApi>;
    this.#logger = createLogger({ type: 'CONSOLE' });
    this.logGroupId = logGroupId;
  }

/**
   * Internal logging helper that formats and sends logs to the configured logger
   * @param level - Severity level of the log
   * @param message - Main log message
   */
 #log (level: LogLevel, message: string) {
    if (!this.#logger) return;

    const logMessage: BaseLogMessage = {
      type: RegisteredLogger.LLM,
      message,
      destinationPath: 'LLM',
      logGroupId: this.logGroupId,
    };

    const logMethod = level.toLowerCase() as keyof Logger<BaseLogMessage>;

    this.#logger[logMethod]?.(logMessage);
  }


  /**
   * Set the concrete tools for the agent
   * @param tools
   */
  __setTools(tools: Record<TKeys, ToolApi>) {
    this.#tools = tools;
     this.#log('DEBUG', `Tools set for LLM`);
  }

   /**
   * Set the logger for the agent
   * @param logger
   */
  __setLogger(logger: Logger) {
    this.#logger = logger;
   this.#log('DEBUG', `Logger updated for LLM`);
  }

  async getModelType(model: ModelConfig): Promise<string> {
    if (!('provider' in model)) {
      throw new Error('Model provider is required');
    }
    const providerToType: Record<LLMProvider, string> = {
      OPEN_AI: 'openai',
      ANTHROPIC: 'anthropic',
      GROQ: 'groq',
      PERPLEXITY: 'perplexity',
      FIREWORKS: 'fireworks',
      TOGETHER_AI: 'togetherai',
      LM_STUDIO: 'lmstuido',
      BASETEN: 'baseten',
      GOOGLE: 'google',
      MISTRAL: 'mistral',
      X_GROK: 'grok',
      COHERE: 'cohere',
      AZURE: 'azure',
      AMAZON: 'amazon',
      //
      ANTHROPIC_VERTEX: 'anthropic-vertex',
    };
    const type =
      providerToType[model.provider as LLMProvider] ?? model.provider;

   this.#log('DEBUG', `Model type resolved to ${type} for provider ${model.provider}`);

    return type;
  }

  createOpenAICompatibleModel({
    baseURL,
    apiKey,
    defaultModelName,
    modelName,
    fetch,
  }: {
    baseURL: string;
    apiKey: string;
    defaultModelName: string;
    modelName?: string;
    fetch?: typeof globalThis.fetch;
  }): LanguageModelV1 {
    this.#log('DEBUG', `Creating OpenAI compatible model with baseURL: ${baseURL}`);
    const client = createOpenAI({
      baseURL,
      apiKey,
      fetch,
    });
    return client(modelName || defaultModelName);
  }

  async createModelDef({
    model,
  }: {
    model: {
      type: string;
      name?: string;
      toolChoice?: 'auto' | 'required';
      baseURL?: string;
      fetch?: typeof globalThis.fetch;
      apiKey?: string;
    };
  }): Promise<LanguageModelV1> {
    let modelDef: LanguageModelV1;
    if (model.type === 'openai') {
      await this.#log('DEBUG', `Initializing OpenAI model ${model.name || 'gpt-4o-2024-08-06'}`);
      const openai = createOpenAI({
        apiKey: model?.apiKey || process.env.OPENAI_API_KEY,
      });
      modelDef = openai(model.name || 'gpt-4o-2024-08-06', {
        structuredOutputs: true,
      });
    } else if (model.type === 'anthropic') {
      await this.#log('DEBUG', `Initializing Anthropic model ${model.name || 'claude-3-5-sonnet-20240620'}`);
      const anthropic = createAnthropic({
        apiKey: model?.apiKey || process.env.ANTHROPIC_API_KEY,
      });
      modelDef = anthropic(model.name || 'claude-3-5-sonnet-20240620');
    } else if (model.type === 'google') {
      await this.#log('DEBUG', `Initializing Google model ${model.name || 'gemini-1.5-pro-latest'}`);
      const google = createGoogleGenerativeAI({
        baseURL: 'https://generativelanguage.googleapis.com/v1beta',
        apiKey: model?.apiKey || process.env.GOOGLE_GENERATIVE_AI_API_KEY || '',
      });
      modelDef = google(model.name || 'gemini-1.5-pro-latest');
    } else if (model.type === 'groq') {
      await this.#log('DEBUG', `Initializing Groq model ${model.name || 'llama-3.2-90b-text-preview'}`);
      modelDef = this.createOpenAICompatibleModel({
        baseURL: 'https://api.groq.com/openai/v1',
        apiKey: model?.apiKey || process.env.GROQ_API_KEY || '',
        defaultModelName: 'llama-3.2-90b-text-preview',
        modelName: model.name,
      });
    } else if (model.type === 'perplexity') {
      await this.#log('DEBUG', `Initializing Perplexity model ${model.name || 'llama-3.1-sonar-large-128k-chat'}`);
      modelDef = this.createOpenAICompatibleModel({
        baseURL: 'https://api.perplexity.ai/',
        apiKey: model?.apiKey || process.env.PERPLEXITY_API_KEY || '',
        defaultModelName: 'llama-3.1-sonar-large-128k-chat',
        modelName: model.name,
      });
    } else if (model.type === 'fireworks') {
      await this.#log('DEBUG', `Initializing Fireworks model ${model.name || 'llama-v3p1-70b-instruct'}`);
      modelDef = this.createOpenAICompatibleModel({
        baseURL: 'https://api.fireworks.ai/inference/v1',
        apiKey: model?.apiKey || process.env.FIREWORKS_API_KEY || '',
        defaultModelName: 'llama-v3p1-70b-instruct',
        modelName: model.name,
      });
    } else if (model.type === 'togetherai') {
      await this.#log('DEBUG', `Initializing TogetherAI model ${model.name || 'google/gemma-2-9b-it'}`);
      modelDef = this.createOpenAICompatibleModel({
        baseURL: 'https://api.together.xyz/v1/',
        apiKey: model?.apiKey || process.env.TOGETHER_AI_API_KEY || '',
        defaultModelName: 'google/gemma-2-9b-it',
        modelName: model.name,
      });
    } else if (model.type === 'lmstudio') {
      await this.#log('DEBUG', `Initializing LMStudio model ${model.name || 'llama-3.2-1b'}`);

      if (!model?.baseURL) {
        const error = `LMStudio model requires a baseURL`;
        await this.#log('ERROR', error);
        throw new Error(error);
      }
      modelDef = this.createOpenAICompatibleModel({
        baseURL: model.baseURL,
        apiKey: 'not-needed',
        defaultModelName: 'llama-3.2-1b',
        modelName: model.name,
      });
    } else if (model.type === 'baseten') {
      await this.#log('DEBUG', `Initializing BaseTen model ${model.name || 'llama-3.1-70b-instruct'}`);
      if (model?.fetch) {
        const error = `Custom fetch is required to use ${model.type}. see https://docs.baseten.co/api-reference/openai for more information`;
        await this.#log('ERROR', error);
        throw new Error(error);
      }
      modelDef = this.createOpenAICompatibleModel({
        baseURL: 'https://bridge.baseten.co/v1/direct',
        apiKey: model?.apiKey || process.env.BASETEN_API_KEY || '',
        defaultModelName: 'llama-3.1-70b-instruct',
        modelName: model.name,
      });
    } else if (model.type === 'mistral') {
      await this.#log('DEBUG', `Initializing Mistral model ${model.name || 'pixtral-large-latest'}`);
      const mistral = createMistral({
        baseURL: 'https://api.mistral.ai/v1',
        apiKey: model?.apiKey || process.env.MISTRAL_API_KEY || '',
      });

      modelDef = mistral(model.name || 'pixtral-large-latest');
    } else if (model.type === 'grok') {
      await this.#log('DEBUG', `Initializing X Grok model ${model.name || 'grok-beta'}`);
      const xAi = createXai({
        baseURL: 'https://api.x.ai/v1',
        apiKey: process.env.XAI_API_KEY ?? '',
      });

      modelDef = xAi(model.name || 'grok-beta');
    } else if (model.type === 'cohere') {
      await this.#log('DEBUG', `Initializing Cohere model ${model.name || 'command-r-plus'}`);
      const cohere = createCohere({
        baseURL: 'https://api.cohere.com/v2',
        apiKey: model?.apiKey || process.env.COHERE_API_KEY || '',
      });

      modelDef = cohere(model.name || 'command-r-plus');
    } else if (model.type === 'azure') {
      await this.#log('DEBUG', `Initializing Azure model ${model.name || 'gpt-35-turbo-instruct'}`);
      const azure = createAzure({
        resourceName: process.env.AZURE_RESOURCE_NAME || '',
        apiKey: model?.apiKey || process.env.AZURE_API_KEY || '',
      });
      modelDef = azure(model.name || 'gpt-35-turbo-instruct');
    } else if (model.type === 'amazon') {
      await this.#log('DEBUG', `Initializing Amazon model ${model.name || 'amazon-titan-tg1-large'}`);
      const amazon = createAmazonBedrock({
        region: process.env.AWS_REGION || '',
        accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
        sessionToken: process.env.AWS_SESSION_TOKEN || '',
      });
      modelDef = amazon(model.name || 'amazon-titan-tg1-large');
    } else if (model.type === 'anthropic-vertex') {
      await this.#log('DEBUG', `Initializing Anthropic Vertex model ${model.name || 'claude-3-5-sonnet@20240620'}`);
      const anthropicVertex = createAnthropicVertex({
        region: process.env.GOOGLE_VERTEX_REGION,
        projectId: process.env.GOOGLE_VERTEX_PROJECT_ID,
        apiKey: process.env.ANTHROPIC_API_KEY ?? '',
      });
      modelDef = anthropicVertex(model.name || 'claude-3-5-sonnet@20240620');
    } else {
      const error = `Invalid model type: ${model.type}`;
      await this.#log('ERROR', error);
      throw new Error(error);
    }

    return modelDef;
  }

  async createEmbedding({
    model,
    value,
    maxRetries,
  }: {
    model: EmbeddingModelConfig;
    value: string[] | string;
    maxRetries: number;
  }) {
    let embeddingModel: EmbeddingModel<string>;

    //yo
    if (model.provider === 'OPEN_AI') {
      const openai = createOpenAI({
        apiKey: process.env.OPENAI_API_KEY,
      });
      embeddingModel = openai.embedding(model.name);
    } else if (model.provider === 'COHERE') {
      const cohere = createCohere({
        apiKey: process.env.COHERE_API_KEY,
      });
      embeddingModel = cohere.embedding(model.name);
    } else {
      throw new Error(`Invalid embedding model`);
    }

    if (value instanceof Array) {
      return await embedMany({
        model: embeddingModel,
        values: value,
        maxRetries,
      });
    }

    return await embed({
      model: embeddingModel,
      value,
      maxRetries,
    });
  }

  async getParams({
    tools,
    resultTool,
    model,
  }: {
    tools: Record<string, CoreTool>;
    resultTool?: { description: string; parameters: ZodSchema };
    model:
      | ({
          type: string;
          name?: string;
          toolChoice?: 'auto' | 'required';
          baseURL?: string;
          apiKey?: string;
          fetch?: typeof globalThis.fetch;
        } & GoogleGenerativeAISettings)
      | CustomModelConfig;
  }) {
    const toolsConverted = Object.entries(tools).reduce(
      (memo, [key, val]) => {
        memo[key] = tool(val);
        return memo;
      },
      {} as Record<string, CT>
    );

    let answerTool = {};
    if (resultTool) {
      answerTool = { answer: tool(resultTool) };
    }

    let modelDef;

    if ('type' in model) {
      modelDef = await this.createModelDef({ model });
    } else {
      if (model.model instanceof Function) {
        modelDef = await model.model();
      } else {
        modelDef = model.model;
      }
    }

    return {
      toolsConverted,
      modelDef,
      answerTool,
      toolChoice: model.toolChoice || 'required',
    };
  }

  convertTools(
    enabledTools?: Partial<Record<TKeys, boolean>>
  ): Record<TKeys, CoreTool> {
    const converted = Object.entries(enabledTools || {}).reduce(
      (memo, value) => {
        const k = value[0] as TKeys;
        const enabled = value[1] as boolean;
        const tool = this.#tools[k];

        if (enabled && tool) {
          memo[k] = {
            description: tool.description,
            parameters: z.object({
              data: tool.schema,
            }),
            execute: tool.executor,
          };
        }
        return memo;
      },
      {} as Record<TKeys, CoreTool>
    );

    this.#log('DEBUG', `Converted tools for LLM`);
    return converted;
  }

  private isBaseOutputType(outputType: StructuredOutputType) {
    return (
      outputType === 'string' ||
      outputType === 'number' ||
      outputType === 'boolean' ||
      outputType === 'date'
    );
  }

  private baseOutputTypeSchema(outputType: StructuredOutputType) {
    switch (outputType) {
      case 'string':
        return z.string();
      case 'number':
        return z.number();
      case 'boolean':
        return z.boolean();
      case 'date':
        return z.string().datetime();
      default:
        return z.string();
    }
  }

  private createOutputSchema(output: StructuredOutput) {
    const schema = Object.entries(output).reduce(
      (memo, [k, v]) => {
        if (this.isBaseOutputType(v.type)) {
          memo[k] = this.baseOutputTypeSchema(v.type);
        }
        if (v.type === 'object') {
          const objectItem = v.items;
          const objectItemSchema = this.createOutputSchema(objectItem);

          memo[k] = objectItemSchema;
        }
        if (v.type === 'array') {
          const arrayItem = v.items;
          if (this.isBaseOutputType(arrayItem.type)) {
            const itemSchema = this.baseOutputTypeSchema(arrayItem.type);
            memo[k] = z.array(itemSchema);
          }

          if (arrayItem.type === 'object') {
            const objectInArrayItemSchema = this.createOutputSchema(
              arrayItem.items
            );
            memo[k] = z.array(objectInArrayItemSchema);
          }
        }
        return memo;
      },
      {} as Record<string, any>
    );

    return z.object(schema);
  }

  async text({
    model,
    messages,
    onStepFinish,
    maxSteps = 5,
    enabledTools,
  }: {
    enabledTools?: Partial<Record<TKeys, boolean>>;
    model: ModelConfig;
    messages: CoreMessage[];
    onStepFinish?: (step: string) => void;
    maxSteps?: number;
  }) {
    let modelToPass;

    if ('name' in model) {
      modelToPass = {
        type: await this.getModelType(model),
        name: model.name,
        toolChoice: model.toolChoice,
        apiKey: model.provider !== 'LM_STUDIO' ? model?.apiKey : undefined,
        baseURL: model.provider === 'LM_STUDIO' ? model.baseURL : undefined,
        fetch: model.provider === 'BASETEN' ? model.fetch : undefined,
      };
    } else {
      modelToPass = model;
    }

    const params = await this.getParams({
      tools: this.convertTools(enabledTools || {}),
      model: modelToPass,
    });

    const argsForExecute = {
      model: params.modelDef,
      tools: {
        ...params.toolsConverted,
        ...params.answerTool,
      },
      toolChoice: params.toolChoice,
      maxSteps,
      onStepFinish: async (props: any) => {
        onStepFinish?.(JSON.stringify(props, null, 2));
        if (
          props?.response?.headers?.['x-ratelimit-remaining-tokens'] &&
          parseInt(
            props?.response?.headers?.['x-ratelimit-remaining-tokens'],
            10
          ) < 2000
        ) {
          this.#log('WARN', 'Rate limit approaching, waiting 10 seconds');
          await delay(10 * 1000);
        }
      },
    };

   this.#log('DEBUG', `Generating text with ${messages.length} messages`);
    return await generateText({
      messages,
      ...argsForExecute,
    });
  }

  async textObject({
    model,
    messages,
    onStepFinish,
    maxSteps = 5,
    enabledTools,
    structuredOutput,
  }: {
    structuredOutput: StructuredOutput;
    enabledTools?: Partial<Record<TKeys, boolean>>;
    model: ModelConfig;
    messages: CoreMessage[];
    onStepFinish?: (step: string) => void;
    maxSteps?: number;
  }) {
    let modelToPass;

    if ('name' in model) {
      modelToPass = {
        type: await this.getModelType(model),
        name: model.name,
        toolChoice: model.toolChoice,
        apiKey: model.provider !== 'LM_STUDIO' ? model?.apiKey : undefined,
        baseURL: model.provider === 'LM_STUDIO' ? model.baseURL : undefined,
        fetch: model.provider === 'BASETEN' ? model.fetch : undefined,
      };
    } else {
      modelToPass = model;
    }

    const params = await this.getParams({
      tools: this.convertTools(enabledTools || {}),
      model: modelToPass,
    });

    const argsForExecute = {
      model: params.modelDef,
      tools: {
        ...params.toolsConverted,
        ...params.answerTool,
      },
      toolChoice: params.toolChoice,
      maxSteps,
      onStepFinish: async (props: any) => {
        onStepFinish?.(JSON.stringify(props, null, 2));
        if (
          props?.response?.headers?.['x-ratelimit-remaining-tokens'] &&
          parseInt(
            props?.response?.headers?.['x-ratelimit-remaining-tokens'],
            10
          ) < 2000
        ) {
          this.#logger.warn('Rate limit approaching, waiting 10 seconds');
          await delay(10 * 1000);
        }
      },
    };

  this.#log('DEBUG', `Generating text with ${messages.length} messages`);

    const schema = this.createOutputSchema(structuredOutput);
    return await generateObject({
      messages,
      ...argsForExecute,
      output: 'object',
      schema,
    });
  }

  async stream({
    model,
    messages,
    onStepFinish,
    onFinish,
    maxSteps = 5,
    enabledTools,
  }: {
    model: ModelConfig;
    enabledTools: Partial<Record<TKeys, boolean>>;
    messages: CoreMessage[];
    onStepFinish?: (step: string) => void;
    onFinish?: (result: string) => Promise<void> | void;
    maxSteps?: number;
  }) {
    let modelToPass;
    if ('name' in model) {
      modelToPass = {
        type: await this.getModelType(model),
        name: model.name,
        toolChoice: model.toolChoice,
        apiKey: model.provider !== 'LM_STUDIO' ? model?.apiKey : undefined,
        baseURL: model.provider === 'LM_STUDIO' ? model.baseURL : undefined,
        fetch: model.provider === 'BASETEN' ? model.fetch : undefined,
      };
    } else {
      modelToPass = model;
    }

    const params = await this.getParams({
      tools: this.convertTools(enabledTools),
      model: modelToPass,
    });

    const argsForExecute = {
      model: params.modelDef,
      tools: {
        ...params.toolsConverted,
        ...params.answerTool,
      },
      toolChoice: params.toolChoice,
      maxSteps,
      onStepFinish: async (props: any) => {
        onStepFinish?.(JSON.stringify(props, null, 2));
        if (
          props?.response?.headers?.['x-ratelimit-remaining-tokens'] &&
          parseInt(
            props?.response?.headers?.['x-ratelimit-remaining-tokens'],
            10
          ) < 2000
        ) {
          this.#log('WARN', 'Rate limit approaching, waiting 10 seconds');
          await delay(10 * 1000);
        }
      },
      onFinish: async (props: any) => {
        onFinish?.(JSON.stringify(props, null, 2));
      },
    };

   this.#log('DEBUG', `Streaming text with ${messages.length} messages`);
    return await streamText({
      messages,
      ...argsForExecute,
    });
  }

  async streamObject({
    model,
    messages,
    onStepFinish,
    onFinish,
    maxSteps = 5,
    enabledTools,
    structuredOutput,
  }: {
    structuredOutput: StructuredOutput;
    model: ModelConfig;
    enabledTools: Partial<Record<TKeys, boolean>>;
    messages: CoreMessage[];
    onStepFinish?: (step: string) => void;
    onFinish?: (result: string) => Promise<void> | void;
    maxSteps?: number;
  }) {
    let modelToPass;
    if ('name' in model) {
      modelToPass = {
        type: await this.getModelType(model),
        name: model.name,
        toolChoice: model.toolChoice,
        apiKey: model.provider !== 'LM_STUDIO' ? model?.apiKey : undefined,
        baseURL: model.provider === 'LM_STUDIO' ? model.baseURL : undefined,
        fetch: model.provider === 'BASETEN' ? model.fetch : undefined,
      };
    } else {
      modelToPass = model;
    }

    const params = await this.getParams({
      tools: this.convertTools(enabledTools),
      model: modelToPass,
    });

    const argsForExecute = {
      model: params.modelDef,
      tools: {
        ...params.toolsConverted,
        ...params.answerTool,
      },
      toolChoice: params.toolChoice,
      maxSteps,
      onStepFinish: async (props: any) => {
        onStepFinish?.(JSON.stringify(props, null, 2));
        if (
          props?.response?.headers?.['x-ratelimit-remaining-tokens'] &&
          parseInt(
            props?.response?.headers?.['x-ratelimit-remaining-tokens'],
            10
          ) < 2000
        ) {
          this.#logger.warn('Rate limit approaching, waiting 10 seconds');
          await delay(10 * 1000);
        }
      },
      onFinish: async (props: any) => {
        onFinish?.(JSON.stringify(props, null, 2));
      },
    };

     this.#log('DEBUG', `Streaming text with ${messages.length} messages`);

    const schema = this.createOutputSchema(structuredOutput);
    return await streamObject({
      messages,
      ...argsForExecute,
      output: 'object',
      schema,
    });
  }
}
