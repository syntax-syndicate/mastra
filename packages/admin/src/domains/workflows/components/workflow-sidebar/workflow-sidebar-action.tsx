import { createId } from '@paralleldrive/cuid2';
import { useEffect, useState } from 'react';

import { ScrollArea } from '@/components/ui/scroll-area';

import { isObjectEmpty } from '@/lib/object';
import { lodashTitleCase } from '@/lib/string';

import { systemLogics } from '../../constants';
import { useWorkflowContext } from '../../context/workflow-context';
import { ActionVariables, AutomationAction, RefinedIntegrationAction, UpdateAutomationAction } from '../../types';
import ActionSelector from '../utils/action-selector';

import DynamicForm from './action-form';
import { DeleteWorkflowActionBlock } from './delete-workflow-action-block';
import { WorkflowSidebarHeader } from './workflow-sidebar-header';

interface WorkflowSidebarActionProps {
  action: AutomationAction;
  blueprintId: string;
}

export function WorkflowSidebarAction({ action, blueprintId }: WorkflowSidebarActionProps) {
  const { updateAction, addNewBlankAction, setSelectedBlock, selectedBlock, actions, frameworkActions } =
    useWorkflowContext();
  const [actionToEdit, setActionToEdit] = useState<null | AutomationAction>(null);
  const parentAction = actions[action.parentActionId || ''];

  function handleEditActionType(action: AutomationAction) {
    setActionToEdit(action);
  }

  function updateActionPayload({
    payload,
    variables,
  }: {
    payload: Record<string, any>;
    variables?: Record<string, ActionVariables>;
  }) {
    const updatePayload: UpdateAutomationAction = {
      ...action,
      payload: action.payload ? { ...action.payload, ...payload } : payload,
      variables: { ...(action.variables || {}), ...(variables || {}) },
    };

    updateAction(updatePayload as any);
    //write to tempt file
  }

  const handleUpdateAction = () => {
    const id = createId();
    addNewBlankAction({
      newAction: { id, type: '', parentActionId: action.id },
    });
    //write to temp file
  };

  const handleCreateAction = (updatedAction: UpdateAutomationAction) => {
    let newAction: UpdateAutomationAction = { ...action, ...updatedAction };
    if (actionToEdit) {
      if (actionToEdit.type === updatedAction.type) {
        setActionToEdit(null);
        return;
      }
      newAction = {
        ...actionToEdit,
        type: updatedAction.type,
        payload: undefined,
      };
    }

    updateAction(newAction as any);

    //write to temp file
    setSelectedBlock({ type: 'action', block: newAction as any });

    setActionToEdit(null);
  };

  const handleBlur = ({
    payload,
    variables,
  }: {
    payload: Record<string, any>;
    variables?: Record<string, ActionVariables>;
  }) => {
    updateActionPayload({ payload, variables });
  };

  useEffect(() => {
    setActionToEdit(null);
  }, [selectedBlock]);

  if (action?.type && !isObjectEmpty(action) && !actionToEdit) {
    return (
      <>
        <WorkflowSidebarHeader
          title="Configure Action"
          type="action"
          onBackToList={() => handleEditActionType(action)}
        />
        {/*this renders the selected action block*/}
        <DynamicForm
          key={action.id}
          action={action}
          onUpdateAction={handleUpdateAction}
          handleEditActionType={handleEditActionType}
          onBlur={handleBlur}
        />
      </>
    );
  }

  const groupByPluginName = frameworkActions?.reduce((acc, fwAct) => {
    return {
      ...acc,
      [fwAct.pluginName]: [...(acc[fwAct.pluginName] || []), fwAct],
    };
  }, {} as { [key: string]: RefinedIntegrationAction[] });

  return (
    <>
      {/*this renders the list of action blocks to select from*/}
      <WorkflowSidebarHeader title={actionToEdit ? 'Change next step' : 'Choose next step'} />
      <ScrollArea>
        <div className="border-kp-border-1 flex flex-col gap-5 border-b-[0.3px] p-6">
          <div className="mb-5 space-y-1">
            <h1 className="text-xs">Actions</h1>
            <p className="text-kp-el-3 text-[11px]">Select an action</p>
          </div>
          <div className="space-y-10">
            {Object.entries(groupByPluginName).map(([pluginName, actionList]) => (
              <div key={pluginName} className="space-y-2">
                <p className="text-xs">{lodashTitleCase(pluginName)} Actions</p>
                {actionList.map(actionItem => (
                  <ActionSelector
                    key={actionItem.type}
                    isSelected={actionToEdit?.type === actionItem.type}
                    type={actionItem.type}
                    onSelectActionEvent={handleCreateAction}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>

        {parentAction?.type === 'CONDITIONS' ? null : (
          <div className="border-kp-border-1 flex flex-col gap-5 border-b-[0.3px] p-6">
            <div className="mb-5 space-y-1">
              <h1 className="text-xs">Logics</h1>
              <p className="text-[11px]">Select a logic</p>
            </div>
            <div className="space-y-2">
              {systemLogics.map(actionItem => (
                <ActionSelector
                  key={actionItem.type}
                  isSelected={actionToEdit?.type === actionItem.type}
                  type={actionItem.type}
                  onSelectActionEvent={handleCreateAction}
                />
              ))}
            </div>
          </div>
        )}

        {action?.id && !action?.type ? (
          <div className="flex justify-end px-6 py-5">
            <DeleteWorkflowActionBlock action={action} deleteOnlyBlock />
          </div>
        ) : null}
      </ScrollArea>
    </>
  );
}