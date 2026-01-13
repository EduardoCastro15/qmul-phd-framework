// Load social media datasets and create graph
LOAD CSV WITH HEADERS FROM "file:///nodes_list.csv" AS line
MERGE (n:Page { nodeId: toInteger(line.id),pageName:line.name, new_id: toInteger(line.new_id) })

LOAD CSV FROM "file:///edges_list.csv" AS line
MATCH (a:Page {new_id: toInteger(line[0])}), (b:Page {new_id: toInteger(line[1])})
MERGE (a)-[:LIKE]->(b)

// Create link prediction pipeline
CALL gds.beta.pipeline.linkPrediction.create('soc_pipe')

// Add node properties (embedding)
CALL gds.beta.pipeline.linkPrediction.addNodeProperty('soc_pipe', 'node2vec', {
    mutateProperty: 'embedding',
    embeddingDimension: 256
})
YIELD nodePropertySteps

// Add link features (HADAMARD)
CALL gds.beta.pipeline.linkPrediction.addFeature('soc_pipe', 'HADAMARD', {
    nodeProperties: ['embedding']
}) YIELD featureSteps

// Add logistic regression
CALL gds.beta.pipeline.linkPrediction.addLogisticRegression('soc_pipe') YIELD parameterSpace

// Create cypher projection
CALL gds.graph.project(
  'soc',
  ['Page'],
  {
    LIKE: {orientation: 'UNDIRECTED'}
  }
)

// Train model
CALL gds.beta.pipeline.linkPrediction.train('soc', {
    modelName: 'soc_model',
    pipeline: 'soc_pipe',
    metrics: ['AUCPR'],
    targetRelationshipType: 'LIKE'
}) YIELD modelInfo, modelSelectionStats
RETURN
modelInfo.bestParameters AS winningModel,
modelInfo.metrics.AUCPR.train.avg AS avgTrainScore,
modelInfo.metrics.AUCPR.outerTrain AS outerTrainScore,
modelInfo.metrics.AUCPR.test AS testScore,
[cand IN modelSelectionStats.modelCandidates | cand.metrics.AUCPR.validation.avg] AS validationScores

// Stream prediction
CALL gds.beta.pipeline.linkPrediction.predict.stream('soc',{
    modelName: 'soc_model',
    sampleRate: 1,
    topN: 40,
    threshold: 0.3
    }
)YIELD node1, node2, probability
 RETURN gds.util.asNode(node1).pageName AS page1, gds.util.asNode(node2).pageName AS page2, probability
 ORDER BY probability DESC

// Drop pipeline, model and projection
 CALL gds.pipeline.drop('soc_pipe');
 CALL gds.model.drop('soc_model');
 CALL gds.graph.drop('soc');

 // Node embedding test
 CALL gds.graph.project(
  'soc',
  ['Page'],
  {
    LIKES: {orientation: 'UNDIRECTED'}
  }
)

CALL gds.node2vec.mutate(
    'soc',
    {
        embeddingDimension: 256,
        mutateProperty: 'embedding'
    }
)YIELD nodeCount, nodePropertiesWritten

// Find similarity of embeddings
CALL gds.knn.stream(
    'soc',
    {
        nodeProperties: ['embedding']
    }
) YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).pageName AS page1, gds.util.asNode(node2).pageName AS page2,  similarity