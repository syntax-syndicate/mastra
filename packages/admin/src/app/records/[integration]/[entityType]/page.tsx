import { future } from '../../../../../example.future.config';
import { ClientLayout } from '.././[entityType]/client-layout';

export default async function Integration({ params }: { params: { integration: string; entityType: string } }) {
  const integrationName = params.integration.toUpperCase();
  const entityType = params.entityType.toUpperCase();
  const integration = future.getIntegration(integrationName);

  if (!integration) {
    console.log(`Integration ${integrationName} not found`);
    return null;
  }
  const referenceId = `1`;

  const connection = await future.dataLayer.getConnectionByReferenceId({
    referenceId,
    name: integrationName,
  });

  if (!connection) {
    console.log(`Connection with referenceId ${referenceId} not found for ${params.integration}`);
    return null;
  }

  const syncTable = await future.dataLayer.getEntityRecordsByConnectionAndType({
    connectionId: connection?.id!,
    type: entityType,
  });

  return (
    <ClientLayout
      integration={params.integration}
      properties={syncTable?.properties || []}
      data={syncTable?.records?.map(({ data }) => data) || []}
      entityTypes={integration.entityTypes}
    />
  );
}