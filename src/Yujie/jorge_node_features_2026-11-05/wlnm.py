"""Jorge WLNM_dir_neg port; not SEAL. See README for protocol differences."""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PRIME_LOGS = np.log(np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                             31, 37, 41, 43, 47, 53, 59, 61, 67, 71], float))


@lru_cache(maxsize=1)
def native_canon():
    lib = ctypes.CDLL(str(ROOT / "libwlnm_nauty.dylib"))
    ptr = ctypes.POINTER(ctypes.c_int)
    lib.wlnm_canon.argtypes = [ctypes.c_int, ptr, ptr, ptr]
    lib.wlnm_canon.restype = ctypes.c_int
    return lib


def tolerance_colors(values):
    """Sorted uniquetol-style classes, MATLAB default double tolerance 1e-12."""
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    tol = 1e-12 * np.max(np.abs(values))
    colors = np.zeros(len(values), dtype=np.int32)
    representative = values[order[0]]
    color = 0
    for i in order:
        if values[i] - representative > tol:
            color += 1
            representative = values[i]
        colors[i] = color
    return colors


def palette_wl(adj, colors):
    colors = np.asarray(colors, dtype=np.int32)
    for _ in range(100):
        logs = PRIME_LOGS[colors]
        signatures = (colors + 1) + adj @ logs / np.ceil(logs.sum())
        refined = tolerance_colors(signatures)
        if np.array_equal(refined, colors):
            return refined
        colors = refined
    raise RuntimeError("WL color refinement did not converge")


def graph_distances(adj, start):
    n = len(adj)
    distances = np.full(n, 2 * n, dtype=int)
    distances[start] = 0
    frontier = [start]
    while frontier:
        nxt = []
        for node in frontier:
            for neighbor in np.flatnonzero(adj[node]):
                if distances[neighbor] == 2 * n:
                    distances[neighbor] = distances[node] + 1
                    nxt.append(int(neighbor))
        frontier = nxt
    return distances


@lru_cache(maxsize=32768)
def canonical_order_cached(n, adjacency_bytes):
    adj = np.frombuffer(adjacency_bytes, np.uint8).reshape(n, n)
    # Preserve upstream graph(subgraph, 'upper'), including its asymmetry.
    upper = np.triu(adj, 1)
    weak_upper = upper + upper.T
    avg = np.sqrt(graph_distances(weak_upper, 0) * graph_distances(weak_upper, 1))
    colors = np.unique(avg, return_inverse=True)[1].astype(np.int32)
    classes = palette_wl(adj, colors)
    sorted_order = np.argsort(classes, kind="stable")
    sorted_adj = np.ascontiguousarray(adj[np.ix_(sorted_order, sorted_order)], np.int32)
    sorted_classes = np.ascontiguousarray(classes[sorted_order], np.int32)
    canon_order = np.empty(n, dtype=np.int32)
    ptr = ctypes.POINTER(ctypes.c_int)
    status = native_canon().wlnm_canon(
        n, sorted_adj.ctypes.data_as(ptr), sorted_classes.ctypes.data_as(ptr),
        canon_order.ctypes.data_as(ptr))
    if status:
        raise RuntimeError(f"nauty status {status}")
    if sorted(canon_order.tolist()) != list(range(n)):
        raise RuntimeError("Invalid canonical permutation")
    return tuple(sorted_order[canon_order].tolist())


def stable_unique(values):
    return list(dict.fromkeys(int(v) for v in values))


class Encoder:
    """Port of graph2vector_dir_neg with candidate masking in both matrices."""
    def __init__(self, adjacency, k=10):
        self.adj = np.asarray(adjacency, dtype=np.uint8)
        self.k = k
        self.successors = [np.flatnonzero(row).tolist() for row in self.adj]
        self.predecessors = [np.flatnonzero(row).tolist() for row in self.adj.T]

    def encode(self, pair, return_debug=False):
        source, target = map(int, pair)
        if source == target:
            raise ValueError("Self pairs are outside this dataset")
        nodes = [source, target]
        links = [(source, target)]
        links_dist = [0]
        dist = 0
        while True:
            dist += 1
            fringe = set()
            for u, v in links:
                fringe.update((u, s) for s in self.successors[u]
                              if (u, s) != (source, target))
                fringe.update((p, v) for p in self.predecessors[v]
                              if (p, v) != (source, target))
            fringe = sorted(fringe)  # MATLAB unique(N, 'rows') order
            if not fringe or dist > 2:
                break
            # MATLAB fringe(:): all first-column nodes, then second column.
            flattened = [x[0] for x in fringe] + [x[1] for x in fringe]
            nodes.extend(x for x in stable_unique(flattened) if x not in nodes)
            links.extend(fringe)
            links_dist.extend([dist] * len(fringe))
            if len(nodes) >= self.k:
                nodes = nodes[:self.k]
                break
        nodes = stable_unique(nodes)
        subgraph = self.adj[np.ix_(nodes, nodes)].copy()
        subgraph[0, 1] = 0
        weights = subgraph.astype(float) / (dist + 1)
        positions = {node: i for i, node in enumerate(nodes)}
        for (u, v), distance in zip(links, links_dist):
            if distance > 0 and u in positions and v in positions:
                weights[positions[u], positions[v]] = 1.0 / distance
        # Also mask the weighted matrix; upstream fails to do this explicitly.
        weights[0, 1] = 0
        order = np.asarray(canonical_order_cached(len(nodes), subgraph.tobytes()))
        ordered = weights[np.ix_(order, order)]
        mask = ~np.eye(len(nodes), dtype=bool)
        vector = ordered.ravel(order="F")[mask.ravel(order="F")]
        # Preserve the upstream epsilon sentinel, even for directed encoding.
        vector[0] = np.finfo(float).eps
        result = np.zeros(self.k * (self.k - 1), dtype=np.float32)
        result[:len(vector)] = vector
        if return_debug:
            return result, {"nodes": nodes, "subgraph": subgraph, "weights": weights,
                            "order": order, "depth_counter": dist}
        return result

    def batch(self, pairs):
        return np.stack([self.encode(pair) for pair in pairs])


def stratified_split(labels, seed, fraction=0.6):
    labels = np.asarray(labels)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("A supervised network needs both labels")
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in [0, 1]:
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 2:
            raise ValueError("Each class needs at least two labels")
        n_train = int(np.floor(fraction * len(indices)))
        train.extend(indices[:n_train])
        test.extend(indices[n_train:])
    return np.asarray(train, int), np.asarray(test, int)


class MLP:
    """90->32->32->16->2, ReLU/softmax, unweighted CE and SGDM.

    NumPy implementation keeps the method independent of SEAL/DGCNN.
    Float32, Glorot-normal weights, zero bias, momentum .9; default shuffle once.
    """
    def __init__(self, feature_dim, seed):
        rng = np.random.default_rng(seed)
        dims = [feature_dim, 32, 32, 16, 2]
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(dims[:-1], dims[1:]):
            scale = np.sqrt(2 / (n_in + n_out))
            self.weights.append(rng.normal(0, scale, (n_in, n_out)).astype(np.float32))
            self.biases.append(np.zeros(n_out, dtype=np.float32))

    def forward(self, x):
        activations = [np.asarray(x, dtype=np.float32)]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            activations.append(np.maximum(activations[-1] @ w + b, 0))
        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs, activations

    def loss_gradients(self, x, y):
        probs, activations = self.forward(x)
        loss = -np.log(np.maximum(probs[np.arange(len(y)), y], 1e-30)).mean()
        delta = probs.copy()
        delta[np.arange(len(y)), y] -= 1
        delta /= len(y)
        dw, db = [None] * 4, [None] * 4
        for i in range(3, -1, -1):
            dw[i] = activations[i].T @ delta
            db[i] = delta.sum(axis=0)
            if i:
                delta = (delta @ self.weights[i].T) * (activations[i] > 0)
        return float(loss), dw, db

    def fit(self, x, y, seed, epochs=200, batch_size=128):
        order = np.random.default_rng(seed).permutation(len(y))
        x = np.asarray(x[order], np.float32)
        y = np.asarray(y[order], int)
        velocity_w = [np.zeros_like(w) for w in self.weights]
        velocity_b = [np.zeros_like(b) for b in self.biases]
        # trainNetwork discards a last partial batch when n >= batch size.
        stop = len(y) if len(y) < batch_size else (len(y) // batch_size) * batch_size
        for epoch in range(epochs):
            lr = 0.1 * (0.9 ** (epoch // 10))
            for start in range(0, stop, batch_size):
                loss, dw, db = self.loss_gradients(x[start:start + batch_size],
                                                   y[start:start + batch_size])
                if not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                for i in range(4):
                    velocity_w[i] *= 0.9
                    velocity_w[i] -= lr * dw[i]
                    velocity_b[i] *= 0.9
                    velocity_b[i] -= lr * db[i]
                    self.weights[i] += velocity_w[i]
                    self.biases[i] += velocity_b[i]
        final_loss = self.loss_gradients(x, y)[0]
        return final_loss

    def predict(self, x):
        p = self.forward(x)[0][:, 1]
        if not np.isfinite(p).all():
            raise FloatingPointError("Nonfinite prediction")
        return p
