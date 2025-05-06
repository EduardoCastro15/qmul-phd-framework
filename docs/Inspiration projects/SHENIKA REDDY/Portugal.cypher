// Loading all food web datasets to create aggregate graph
LOAD CSV WITH HEADERS FROM
'file:///Portugal_L1P2.csv' as row

MERGE (c:Consumer{taxonomy:row.conTaxonomy})
SET
c.taxonomy = row.conTaxonomy,
c.foodweb = row.foodwebName,
c.mass = toFloat(row.conMassMean)

MERGE (r:Resource{taxonomy:row.resTaxonomy})
SET
r.taxonomy = row.resTaxonomy,
r.foodweb = row.foodwebName, 
r.mass = toFloat(row.resMassMean);

LOAD CSV WITH HEADERS FROM
'file:///Portugal_L1P2.csv' as row

WITH row
MATCH (source:Consumer{taxonomy:row.conTaxonomy})
MATCH (target:Resource{taxonomy:row.resTaxonomy})
MERGE (source)-[i: INTERACTS_WITH]->(target)
SET
i.interactionType = row.interactionType,
i.ID = toInteger(trim(row.autoID)),
i.bmr = toFloat(row.bodyMassRatio);

LOAD CSV WITH HEADERS FROM
'file:///Portugal_CR1P3.csv' as row

MERGE (c:Consumer{taxonomy:row.conTaxonomy})
SET
c.taxonomy = row.conTaxonomy,
c.foodweb = row.foodwebName,
c.mass = toFloat(row.conMassMean)

MERGE (r:Resource{taxonomy:row.resTaxonomy})
SET
r.taxonomy = row.resTaxonomy,
r.foodweb = row.foodwebName, 
r.mass = toFloat(row.resMassMean);

LOAD CSV WITH HEADERS FROM
'file:///Portugal_CR1P3.csv' as row

WITH row
MATCH (source:Consumer{taxonomy:row.conTaxonomy})
MATCH (target:Resource{taxonomy:row.resTaxonomy})
MERGE (source)-[i: INTERACTS_WITH]->(target)
SET
i.interactionType = row.interactionType,
i.ID = toInteger(trim(row.autoID)),
i.bmr = toFloat(row.bodyMassRatio);

LOAD CSV WITH HEADERS FROM
'file:///Portugal_CR2P4.csv' as row

MERGE (c:Consumer{taxonomy:row.conTaxonomy})
SET
c.taxonomy = row.conTaxonomy,
c.foodweb = row.foodwebName,
c.mass = toFloat(row.conMassMean)

MERGE (r:Resource{taxonomy:row.resTaxonomy})
SET
r.taxonomy = row.resTaxonomy,
r.foodweb = row.foodwebName, 
r.mass = toFloat(row.resMassMean);

LOAD CSV WITH HEADERS FROM
'file:///Portugal_CR2P4.csv' as row

WITH row
MATCH (source:Consumer{taxonomy:row.conTaxonomy})
MATCH (target:Resource{taxonomy:row.resTaxonomy})
MERGE (source)-[i: INTERACTS_WITH]->(target)
SET
i.interactionType = row.interactionType,
i.ID = toInteger(trim(row.autoID)),
i.bmr = toFloat(row.bodyMassRatio);

LOAD CSV WITH HEADERS FROM
'file:///Portugal_L2P1.csv' as row

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

