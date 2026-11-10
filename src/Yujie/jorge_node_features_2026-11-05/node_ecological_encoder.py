"""Keep Jorge topology/canonicalization intact; align attributes to its order."""
import numpy as np
from wlnm import Encoder


def ordered_subgraphs(encoder,pairs):
    topology=[];node_ids=[]
    for pair in pairs:
        z,debug=encoder.encode(pair,return_debug=True)
        nodes=np.asarray(debug['nodes'],int)[debug['order']]
        assert len(nodes)<=encoder.k and len(np.unique(nodes))==len(nodes)
        padded=np.full(encoder.k,-1,np.int16);padded[:len(nodes)]=nodes
        topology.append(z);node_ids.append(padded)
    return np.asarray(topology,np.float32),np.asarray(node_ids,np.int16)


def fit_node_preprocessor(numeric,categories,train_nodes):
    ids=np.unique(np.asarray(train_nodes,int))
    assert ids.ndim==1 and (ids>=0).all()
    x=np.asarray(numeric,float)[ids];cats=np.asarray(categories,str)[ids]
    medians=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in x.T])
    filled=np.where(np.isfinite(x),x,medians);centers=filled.mean(axis=0);scales=filled.std(axis=0)
    scales=np.where(scales>1e-12,scales,1.)
    return dict(training_node_ids=ids.tolist(),medians=medians.tolist(),centers=centers.tolist(),scales=scales.tolist(),categories=np.unique(cats).tolist())


def transform_nodes(numeric,categories,state):
    x=np.asarray(numeric,float);cats=np.asarray(categories,str)
    missing=~np.isfinite(x)
    values=(np.where(missing,np.asarray(state['medians']),x)-np.asarray(state['centers']))/np.asarray(state['scales'])
    onehot=(cats[:,None]==np.asarray(state['categories'])[None,:]).astype(float)
    # A validity channel distinguishes a real node from an all-zero padding slot.
    return np.column_stack([values,missing.astype(float),onehot,np.ones(len(x))]).astype(np.float32)


def aligned_node_block(ordered_ids,node_features):
    ids=np.asarray(ordered_ids,int)
    block=np.zeros((*ids.shape,node_features.shape[1]),np.float32)
    valid=ids>=0;block[valid]=node_features[ids[valid]]
    assert not block[~valid].any()
    return block.reshape(len(ids),-1)
