'use client';

import { useEffect, useState } from 'react';

import IconButton from '@/components/ui/icon-button';

import { useCopyToClipboard } from '@/lib/hooks/useCopyToClipboard';

import { Icon } from '@/app/components/icon';

import { codeToHtml } from 'shiki/bundle/web';

import { useEventPlaygroundContext } from '../providers/event-playground-provider';

function EventDetail() {
  const { selectedEvent, payload } = useEventPlaygroundContext();
  const selectedEventPlugin = selectedEvent?.integrationName;
  const [codeBlock, setCodeBlock] = useState<string | null>(null);
  const [_, CopyFn, isCodeBlockCopied] = useCopyToClipboard();
  const [snippet, setSnippet] = useState<string>('');

  useEffect(() => {
    if (!selectedEvent || !selectedEventPlugin) {
      return;
    }

    const stringifiedPayload = JSON.stringify(payload, null, 10);

    const snippet = `

        import frameworkInstance from 'path-to-framework-instance';


         frameworkInstance.triggerSystemEvent({
          name: 'workflow/run-automations',
          data: {
            trigger: '${selectedEvent.type}',
            payload: {
              ${stringifiedPayload.substring(1, stringifiedPayload.length - 1)}
            },
          },
          user: {
            referenceId: 1,
          },
        });

      `;

    const getCodeBlock = async () => {
      const html = await codeToHtml(snippet, {
        theme: 'vitesse-dark',
        lang: 'ts',
      });

      return html;
    };

    setSnippet(snippet);
    getCodeBlock().then(html => setCodeBlock(html));
  }, [selectedEvent, selectedEventPlugin, payload]);

  return selectedEvent ? (
    <div className="px-8 h-full grid place-items-center max-w-full overflow-auto">
      <div className="w-full h-max relative group">
        <IconButton
          onClick={() => CopyFn(snippet)}
          variant={'secondary'}
          className="absolute top-4 right-4 w-8 h-8 p-0 opacity-0 group-hover:opacity-100 transition-opacity duration-150 ease-in-out"
        >
          {isCodeBlockCopied ? (
            <Icon name="check" className="text-white" />
          ) : (
            <Icon name="clipboard" className="text-white" />
          )}
        </IconButton>
        <div
          dangerouslySetInnerHTML={{
            __html: codeBlock || '',
          }}
        />
      </div>
    </div>
  ) : (
    <></>
  );
}

export default EventDetail;