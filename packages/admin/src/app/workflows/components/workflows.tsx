'use client';

import React from 'react';

import { useManageWorkflow } from '../hooks/use-manage-workflow';

import WorkflowsTable from './workflows-table';
import { workflowsColumns } from './workflows-table-columns';

export const Workflows = () => {
  const { handleOpenWorkflow, handleRunWorkflow, handleDeleteWorkflow } = useManageWorkflow();
  const workflows = [
    {
      id: 'clz8mdxyd00534qm09leywj1s',
      title: 'New Workflow',
      description: null,
      status: 'DRAFT',
      actions: [
        {
          id: 's3rtwvx51bxjefyq05kuto83',
          type: '',
          subActions: [
            {
              id: 'rxhxtqe35x29iwvji6tug3w5',
              type: 'CREATE_RECORD',
              subActions: [],
              parentActionId: 's3rtwvx51bxjefyq05kuto83',
            },
          ],
        },
      ],
      trigger: {
        id: 'jj6qoqs2fznkdkhns9lgge5i',
        type: 'RECORD_UPDATED',
        payload: {
          value: {
            recordType: 'companies',
          },
        },
      },
      createdAt: '2024-07-30T16:15:59.270Z',
      updatedAt: '2024-07-30T16:33:15.260Z',
      ownerId: 'clvktzwg2000012d2bxppw5em',
      workspaceId: 'clvktzwhf000212d2of5h3xvj',
      owner: {
        id: 'clvktzwg2000012d2bxppw5em',
        authUserId: 'ea2c5450-11c5-40e9-947a-44983cbf1f78',
        photoURL: null,
        firstName: 'Joshua',
        lastName: 'Folorunsho',
        email: 'joshua@getkepler.com',
        createdAt: '2024-04-29T10:43:25.827Z',
        updatedAt: '2024-04-29T10:43:25.827Z',
        activeWorkspaceId: 'clvktzwhf000212d2of5h3xvj',
        assistantThreadId: null,
      },
      runs: [],
    },
    {
      id: 'clz1i01kf000ldvqujk0idrp1',
      title: 'New Workflow',
      description: null,
      status: 'PUBLISHED',
      actions: [
        {
          id: 't6bxipp3xztc1jx44yx1b0zj',
          type: 'SEND_MESSAGE_TO_CHANNEL',
          payload: {
            message: 'A deal has been created from automation',
            channelId: 'C06CRL8187L',
          },
          variables: {},
          subActions: [
            {
              id: 'f3kfuwmhasjm4p9ateypg6aa',
              type: 'SEND_EMAIL',
              payload: {
                to: ['joshuafolorunsho01@gmail.com'],
                body: 'LFG!',
                emailId: '',
                subject: 'LFG!',
              },
              variables: {},
              subActions: [],
              parentActionId: 't6bxipp3xztc1jx44yx1b0zj',
            },
          ],
        },
      ],
      trigger: {
        id: 'ymb8endqno3hi77z9cgzcpd8',
        type: 'RECORD_CREATED',
        payload: {
          value: {
            recordType: 'deals',
          },
        },
      },
      createdAt: '2024-07-25T16:38:49.071Z',
      updatedAt: '2024-07-25T16:55:26.259Z',
      ownerId: 'clvktzwg2000012d2bxppw5em',
      workspaceId: 'clvktzwhf000212d2of5h3xvj',
      owner: {
        id: 'clvktzwg2000012d2bxppw5em',
        authUserId: 'ea2c5450-11c5-40e9-947a-44983cbf1f78',
        photoURL: null,
        firstName: 'Joshua',
        lastName: 'Folorunsho',
        email: 'joshua@getkepler.com',
        createdAt: '2024-04-29T10:43:25.827Z',
        updatedAt: '2024-04-29T10:43:25.827Z',
        activeWorkspaceId: 'clvktzwhf000212d2of5h3xvj',
        assistantThreadId: null,
      },
      runs: [],
    },
    {
      id: 'clz1hfn6n0001hm3x0usukk7l',
      title: 'New Workflow',
      description: null,
      status: 'PUBLISHED',
      actions: [
        {
          id: 'h9hx4wzab6lyn46u278ft77r',
          type: 'CREATE_NOTE',
          payload: {
            title: 'Note 11111',
          },
          variables: {},
          subActions: [],
        },
      ],
      trigger: {
        id: 'odha6fdo8z221xw32ya5m7mn',
        type: 'RECORD_CREATED',
        payload: {
          value: {
            recordType: 'companies',
          },
        },
        condition: {
          or: [
            {
              id: 'rxvyn10lg68uyu5lcts85j32',
              field: '',
              value: '',
              operator: '',
            },
          ],
          field: 'data.email',
          value: '',
          operator: '',
        },
      },
      createdAt: '2024-07-25T16:22:57.312Z',
      updatedAt: '2024-07-25T16:36:05.688Z',
      ownerId: 'clvktzwg2000012d2bxppw5em',
      workspaceId: 'clvktzwhf000212d2of5h3xvj',
      owner: {
        id: 'clvktzwg2000012d2bxppw5em',
        authUserId: 'ea2c5450-11c5-40e9-947a-44983cbf1f78',
        photoURL: null,
        firstName: 'Joshua',
        lastName: 'Folorunsho',
        email: 'joshua@getkepler.com',
        createdAt: '2024-04-29T10:43:25.827Z',
        updatedAt: '2024-04-29T10:43:25.827Z',
        activeWorkspaceId: 'clvktzwhf000212d2of5h3xvj',
        assistantThreadId: null,
      },
      runs: [],
    },
  ];

  return (
    <WorkflowsTable
      data={workflows}
      columns={workflowsColumns({
        handleOpenWorkflow,
        handleRunWorkflow,
        handleDeleteWorkflow,
      })}
    />
  );
};