// Load dataset and create graph
LOAD CSV WITH HEADERS FROM
'file:///Weddell_Sea.csv' as row

MERGE (c:Consumer{taxonomy:row.conTaxonomy})
SET
c.taxonomy = row.conTaxonomy,
c.foodweb = row.foodwebName,
c.mass = toFloat(row.conMassMean)

MERGE (r:Resource{taxonomy:row.resTaxonomy})
SET
r.taxonomy = row.resTaxonomy,
r.foodweb = row.foodwebName, 
r.mass = toFloat(row.resMassMean)

WITH row
MATCH (source:Consumer{taxonomy:row.conTaxonomy})
MATCH (target:Resource{taxonomy:row.resTaxonomy})
MERGE (source)-[i: INTERACTS_WITH]->(target)
SET
i.interactionType = row.interactionType,
i.ID = toInteger(trim(row.autoID))

// Create cypher projection
CALL gds.graph.project(
  'WeddellSea',
  {
    Consumer: {properties: 'mass'},
    Resource: {properties: 'mass'}
  },
  {
    INTERACTS_WITH: {orientation: 'UNDIRECTED'}
  }
)

// Run degree centrality algorithm
CALL gds.degree.stream(
    'WeddellSea'
)
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).taxonomy AS species,labels(gds.util.asNode(nodeId)) AS nodeLabels, score AS score
ORDER BY score DESC 
LIMIT 15

// Run k-nearest neighbours algorithm
CALL gds.knn.stream(
    'WeddellSea',
    {
        nodeProperties: ['mass']
    }
)
YIELD node1, node2, similarity
WITH gds.util.asNode(node1) AS node1, gds.util.asNode(node2)AS node2, similarity
MATCH (node1)-[i:INTERACTS_WITH]->(node2)
SET i.dissimilarity = 1-similarity
RETURN node1.taxonomy as node1, node2.taxonomy as node2, similarity, i.dissimilarity as inverse

// Create link prediction pipeline
CALL gds.beta.pipeline.linkPrediction.create('WeddellSea_pipe')

// Add node properties (embedding)
CALL gds.beta.pipeline.linkPrediction.addNodeProperty('WeddellSea_pipe', 'node2vec', {
    mutateProperty: 'embedding',
    embeddingDimension: 128
})
YIELD nodePropertySteps

// Add link features (HADAMARD)
CALL gds.beta.pipeline.linkPrediction.addFeature('WeddellSea_pipe', 'HADAMARD', {
    nodeProperties: ['embedding', 'mass']
}) YIELD featureSteps
 
 // Add logistic regression
CALL gds.beta.pipeline.linkPrediction.addLogisticRegression('WeddellSea_pipe') YIELD parameterSpace

// Train model
CALL gds.beta.pipeline.linkPrediction.train('WeddellSea', {
    modelName: 'WeddellSea_model',
    pipeline: 'WeddellSea_pipe',
    metrics: ['AUCPR'],
    targetRelationshipType: 'INTERACTS_WITH'
}) YIELD modelInfo, modelSelectionStats
RETURN
modelInfo.bestParameters AS winningModel,
modelInfo.metrics.AUCPR.train.avg AS avgTrainScore,
modelInfo.metrics.AUCPR.outerTrain AS outerTrainScore,
modelInfo.metrics.AUCPR.test AS testScore,
[cand IN modelSelectionStats.modelCandidates | cand.metrics.AUCPR.validation.avg] AS validationScores

// Stream prediction
CALL gds.beta.pipeline.linkPrediction.predict.stream('WeddellSea',{
    modelName: 'WeddellSea_model',
    sampleRate: 1,
    topN: 40,
    threshold: 0.3
    }
)YIELD node1, node2, probability
 RETURN gds.util.asNode(node1).taxonomy AS species1, gds.util.asNode(node2).taxonomy AS species2, probability
 ORDER BY probability DESC

// Drop pipeline, model and projection
 CALL gds.pipeline.drop('WeddellSea_pipe');
 CALL gds.model.drop('WeddellSea_model');
 CALL gds.graph.drop('WeddellSea');

// Node embedding test
 CALL gds.graph.project(
  'WeddellSea',
  {
    Consumer: {properties: 'mass'},
    Resource: {properties: 'mass'}
  },
  {
    INTERACTS_WITH: {orientation: 'UNDIRECTED'}
  }
)

CALL gds.node2vec.mutate(
    'WeddellSea',
    {
        embeddingDimension: 128,
        mutateProperty: 'embedding'
    }
)YIELD nodeCount, nodePropertiesWritten

// Find similarity of embeddings
CALL gds.knn.stream(
    'WeddellSea',
    {
        nodeProperties: ['embedding']
    }
) YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).taxonomy AS species1, gds.util.asNode(node2).taxonomy AS species2,  similarity


