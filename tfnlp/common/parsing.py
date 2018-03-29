# Copyright 2016 Timothy Dozat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def find_cycles(edges):
    vertices = np.arange(len(edges))
    indices = np.zeros_like(vertices) - 1
    lowlinks = np.zeros_like(vertices) - 1
    stack = []
    onstack = np.zeros_like(vertices, dtype=np.bool)
    current_index = 0
    cycles = []

    # ===========================================================================
    def strong_connect(_vertex, _current_index):

        indices[_vertex] = _current_index
        lowlinks[_vertex] = _current_index
        stack.append(_vertex)
        _current_index += 1
        onstack[_vertex] = True

        for vertex_ in np.where(edges == _vertex)[0]:
            if indices[vertex_] == -1:
                _current_index = strong_connect(vertex_, _current_index)
                lowlinks[_vertex] = min(lowlinks[_vertex], lowlinks[vertex_])
            elif onstack[vertex_]:
                lowlinks[_vertex] = min(lowlinks[_vertex], indices[vertex_])

        if lowlinks[_vertex] == indices[_vertex]:
            cycle = []
            vertex_ = -1
            while vertex_ != _vertex:
                vertex_ = stack.pop()
                onstack[vertex_] = False
                cycle.append(vertex_)
            if len(cycle) > 1:
                cycles.append(np.array(cycle))
        return _current_index

    # ===========================================================================

    for vertex in vertices:
        if indices[vertex] == -1:
            current_index = strong_connect(vertex, current_index)
    return cycles


def find_roots(edges):
    return np.where(edges[1:] == 0)[0] + 1


def make_root(probs, root):
    probs = np.array(probs)
    probs[1:, 0] = 0
    probs[root, :] = 0
    probs[root, 0] = 1
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs


def greedy(probs):
    edges = np.argmax(probs, axis=1)
    cycles = True
    while cycles:
        cycles = find_cycles(edges)
        for cycle_vertices in cycles:
            # Get the best heads and their probabilities
            cycle_edges = edges[cycle_vertices]
            cycle_probs = probs[cycle_vertices, cycle_edges]
            # Get the second-best edges and their probabilities
            probs[cycle_vertices, cycle_edges] = 0
            backoff_edges = np.argmax(probs[cycle_vertices], axis=1)
            backoff_probs = probs[cycle_vertices, backoff_edges]
            probs[cycle_vertices, cycle_edges] = cycle_probs
            # Find the node in the cycle that the model is the least confident about and its probability
            new_root_in_cycle = np.argmax(backoff_probs / cycle_probs)
            new_cycle_root = cycle_vertices[new_root_in_cycle]
            # Set the new root
            probs[new_cycle_root, cycle_edges[new_root_in_cycle]] = 0
            edges[new_cycle_root] = backoff_edges[new_root_in_cycle]
    return edges


def score_edges(probs, edges):
    return np.sum(np.log(probs[np.arange(1, len(probs)), edges[1:]]))


def nonprojective(probs):
    probs *= 1 - np.eye(len(probs)).astype(np.float32)
    # ensure head/dummy token is assigned itself as a head
    probs[0] = 0
    probs[0, 0] = 1
    probs /= np.sum(probs, axis=1, keepdims=True)

    edges = greedy(probs)
    roots = find_roots(edges)
    best_edges = edges
    best_score = -np.inf
    if len(roots) > 1:
        for root in roots:
            _probs = make_root(probs, root)
            _edges = greedy(_probs)
            score = score_edges(_probs, _edges)
            if score > best_score:
                best_edges = _edges
                best_score = score
    return best_edges
