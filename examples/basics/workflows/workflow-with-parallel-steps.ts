import { Step, Workflow } from '@mastra/core';
import { z } from 'zod';

const stepOne = new Step({
  id: 'stepOne',
  inputSchema: z.object({
    inputValue: z.number(),
  }),
  outputSchema: z.object({
    doubledValue: z.number(),
  }),
  execute: async ({ context }) => {
    const doubledValue = context.inputValue * 2;
    return { doubledValue };
  },
});

const stepTwo = new Step({
  id: 'stepTwo',
  inputSchema: z.object({
    valueToIncrement: z.number(),
  }),
  outputSchema: z.object({
    incrementedValue: z.number(),
  }),
  execute: async ({ context }) => {
    const incrementedValue = context.valueToIncrement + 1;
    return { incrementedValue };
  },
});

// Build the workflow
const myWorkflow = new Workflow({
  name: 'my-workflow',
  triggerSchema: z.object({
    inputValue: z.number(),
  }),
});

// run workflows in parallel
myWorkflow
  .step(stepOne, {
    variables: {
      inputValue: {
        step: 'trigger',
        path: 'inputValue',
      },
    },
  })
  .step(stepTwo, {
    variables: {
      valueToIncrement: {
        step: 'trigger',
        path: 'inputValue',
      },
    },
  });
myWorkflow.commit();

const res = await myWorkflow.execute({ triggerData: { inputValue: 90 } });
console.log(res.results);
