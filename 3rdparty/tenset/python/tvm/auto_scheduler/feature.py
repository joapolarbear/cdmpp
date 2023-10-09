# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""""
Python API for Feature extraction. The extracted features vector are used by cost models.

We extract one feature vector per BufferStoreNode statement in a TIR Stmt,
so we call this feature as "per-store" feature.
The cost model also does prediction for each BufferStoreNode statement and aggregates
the predicted score of each BufferStoreNode as the score of a TIR Stmt.

The feature specification is defined by `src/auto_scheduler/feature.cc::FeatureSet`
"""

from typing import List, Tuple, Union, Optional
import struct

import numpy as np
import os

from .loop_state import State, StateObject
from .measure import MeasureInput, MeasureResult
from . import _ffi_api

# The maximum number of extracted buffers for one statement
DEFAULT_MAX_N_BUFS = 5

# The length of the feature vector
DEFAULT_FEATURE_VEC_LEN = 164

# The size of int and float in bytes
SIZE_OF_INT32 = 4
SIZE_OF_FLOAT32 = 4

# Whether to extract AST
DEFAULT_PARSE_AST = False
DEBUG_AST = os.environ.get("DEBUG_AST", "0")
DEBUG_AST = True if DEBUG_AST == "1" else False

def _unpack_basic_feature(size, vec_len, byte_arr, offset, parse_ast=False):
    ''' Unpack the features for one record or one leaf node if AST is parsed 

    # Now, we need to unpack the feature for multiple statements/stages.
    # The format is:
    # {
    #   int   n_stage;                        // The number of stages
    #   float feature_vecs[n_stage][vec_len]  // The feature vector for each stage
    #   int   node_id; (Optional)             //  Node_id, only available when parse_ast is True
    # }
    # where vec_len can be calculated by `(size - 1) / n_stages` or `(size - 2) / n_stages`

    Parameters
    ----------
    size: int
        The size of the corresponding data, in the number of floats
    vec_len: int
        Length of features for each stage
    byte_arr: bytearray
        The two-dimensional feature vector in serialized byte array format
    offset: int
        Offset of `byte_arr` in bytes
    parse_ast: bool
        Parse AST if set True
    
    Returns
    -------
    offset: int
        Updated offset
    ret_features: List[List[float]]
        Returned features, shape = [N_stages, vec_len]
    node_id: int (Optional)
        If parse_ast is set True, return the node id of current node
    '''
    stages_by_entries = []
    if size == 0:
        # failed during lowering
        ret_features = np.zeros((1, vec_len))
    else:
        n_stages = struct.unpack_from("f", byte_arr, offset=offset)
        offset += SIZE_OF_FLOAT32

        n_stages = int(n_stages[0] + 0.5)
        tmp_vec_len = (size - 2) // n_stages if parse_ast else (size - 1) // n_stages
        assert (
            tmp_vec_len == vec_len
        ), "The length of feature vector is wrong. Expected %d but got %d." % (
            vec_len,
            tmp_vec_len,
        )
        if parse_ast:
            assert tmp_vec_len * n_stages == size - 2
        else:
            assert tmp_vec_len * n_stages == size - 1
        for _ in range(n_stages):
            x = struct.unpack_from("%df" % vec_len, byte_arr, offset=offset)
            offset += vec_len * SIZE_OF_FLOAT32
            stages_by_entries.append(x)
        ret_features = np.array(stages_by_entries)

    if parse_ast:
        ### Parse node_id
        node_id = struct.unpack_from("f", byte_arr, offset=offset)
        offset += SIZE_OF_FLOAT32
        node_id = int(node_id[0] + 0.5)
    else:
        node_id = -1

    return offset, ret_features, node_id

def unpack_feature(byte_arr: bytearray, parse_ast: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack the flatten feature (in byte array format) from c++

    Parameters
    ----------
    byte_arr: bytearray
        The two-dimensional feature vector in serialized byte array format
    parse_ast: bool
        If set True, parse AST+features

    Returns
    -------
    features: np.ndarray
        Feature vectors
    normalized_throughputs: np.ndarray
        Normalized throughputs
    task_ids: np.ndarray
        Task ids
    min_latency: np.ndarray
        Minimal latency for tasks

    Note
    ----
    For faster data copy between c++ and python, the c++ part returns features in a single
    flatten array using a packed format. The python part then unpacks the flatten array.

    The packed format for n records is:
    {
      int   n;
      int   sizes[n+3];           // The sizes for the following arrays

      float features_0[size[0]];  // The features for record 0
      float features_1[size[1]];  // The features for record 1
      ...
      float features_i[size[i]];  // The features for record i
      ... // until i == n - 1

      float throughputs[sizes[n]];  // The normalized throughputs for n records
      int   task_ids[size[n+1]];    // The task ids for n records
      float min_costs[size[n+2]];   // The min costs for all tasks
    }
    To implement this format, we also store int as float, so we can store all numbers
    into a single float array.
    """
    vec_len = DEFAULT_FEATURE_VEC_LEN

    # unpack sizes
    offset = 0
    n = struct.unpack_from("1i", byte_arr, offset=offset)[0]
    offset += SIZE_OF_INT32

    sizes = struct.unpack_from("%di" % (n + 3), byte_arr, offset=offset)
    offset += SIZE_OF_INT32 * (n + 3)

    # unpack features
    features = []
    for size in sizes[:-3]:
        # For each record
        if parse_ast:
            if size == 0:
                if DEBUG_AST:
                    print("Failed during lowering")
                # failed during lowering
                features.append((np.zeros((1, vec_len)), [], []))
                continue
            # Now, we need to unpack the AST+feature.
            # The format is:
            # {
            #   float   n_leaf;                                 // The number of leaf nodes
            #   float   serialized_ast_size;                    // The size of serialized_ast_size
            #   float   serialized_tree[serialized_ast_size];    // Serialized AST, follow the link https://www.geeksforgeeks.org/serialize-deserialize-n-ary-tree/
            #   float   feature_vecs[n_leaf][leaf_node_feature_size] // The feature vector for each leaf_node
            # }
            n_leaf = struct.unpack_from("f", byte_arr, offset=offset)
            offset += SIZE_OF_FLOAT32
            n_leaf = int(n_leaf[0] + 0.5)

            serialized_ast_size = struct.unpack_from("f", byte_arr, offset=offset)
            offset += SIZE_OF_FLOAT32
            serialized_ast_size = int(serialized_ast_size[0] + 0.5)

            serialized_tree = struct.unpack_from("%df" % serialized_ast_size, byte_arr, offset=offset)
            offset += serialized_ast_size * SIZE_OF_FLOAT32
            serialized_tree = [int(x) for x in serialized_tree]
            
            if DEBUG_AST:
                print("\n\n############# Python Level Log ############")
                print(f"n_leaf: {n_leaf}")
                print(f"serialized_ast_size: {serialized_ast_size}")
                print(f"serialized_tree: {serialized_tree}")

            leaf_node_feature_size = (size - (1 + 1 + serialized_ast_size)) // n_leaf
            ast_features = []
            node_ids = []
            for _ in range(n_leaf):
                offset, ret_features, node_id = _unpack_basic_feature(
                    leaf_node_feature_size,
                    vec_len,
                    byte_arr,
                    offset,
                    parse_ast=True)
                ast_features.append(ret_features)
                node_ids.append(node_id)
                if DEBUG_AST:
                    print(f"Node {node_id}'s feature size: {np.array(ret_features).shape}")
            features.append((
                np.array(ast_features),
                np.array(node_ids),
                np.array(serialized_tree)))
        else:
            ### In this case, features.shape = []
            offset, ret_features, _ = _unpack_basic_feature(
                size, vec_len, byte_arr, offset)
            features.append(ret_features)

    # unpack normalized_throughputs
    m = sizes[-3]
    normalized_throughputs = struct.unpack_from("%df" % m, byte_arr, offset=offset)
    offset += m * SIZE_OF_FLOAT32

    # unpack task_ids
    m = sizes[-2]
    task_ids = struct.unpack_from("%di" % m, byte_arr, offset=offset)
    offset += m * SIZE_OF_INT32

    # unpack min_costs
    m = sizes[-1]
    min_costs = struct.unpack_from("%df" % m, byte_arr, offset=offset)
    offset += m * SIZE_OF_FLOAT32

    assert offset == len(byte_arr), "%d vs %d" % (offset, len(byte_arr))
    return (
        np.array(features, dtype=object),
        np.array(normalized_throughputs),
        np.array(task_ids),
        np.array(min_costs),
    )

def get_per_store_features_from_file(
    filename: str, max_lines: int, max_n_bufs: Optional[int] = None,
    parse_ast: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get per-store features from a log file

    Parameters
    ----------
    filename: str
        The input filename
    max_lines: int
        Only extract the first n lines of the file
    max_n_bufs: Optional[int]
        The maximum number of extracted buffers for one statement

    Returns
    -------
    features: np.ndarray
        Feature vectors
    normalized_throughputs: np.ndarray
        Normalized throughputs
    task_ids: np.ndarray
        Task ids
    min_latency: np.ndarray
            Minimal latency for tasks
    parse_ast: bool
        Reture AST+features if set True
    """
    byte_arr = _ffi_api.GetPerStoreFeaturesFromFile(
        filename, max_lines, max_n_bufs or DEFAULT_MAX_N_BUFS,
        parse_ast or DEFAULT_PARSE_AST
    )
    return unpack_feature(byte_arr, parse_ast)


def get_per_store_features_from_measure_pairs(
    inputs: List[MeasureInput],
    results: List[MeasureResult],
    skip_first_n_feature_extraction: int = 0,
    max_n_bufs: Optional[int] = None,
    parse_ast: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get per-store features from measurement input/result pairs

    Parameters
    ----------
    inputs: List[MeasureInput]
        The measure inputs
    results: List[MeasureResult]
        The measure results
    skip_first_n_feature_extraction: int
        Skip feature extraction for the first n states
    max_n_bufs: int
        The maximum number of extracted buffers for one statement
    parse_ast: bool
        Reture AST+features if set True

    Returns
    -------
    features: np.ndarray
        Feature vectors
    normalized_throughputs: np.ndarray
        Normalized throughputs
    task_ids: np.ndarray
        Task ids
    min_latency: np.ndarray
        Minimal latency for tasks
    """
    byte_arr = _ffi_api.GetPerStoreFeaturesFromMeasurePairs(
        inputs, results, skip_first_n_feature_extraction, max_n_bufs or DEFAULT_MAX_N_BUFS,
        parse_ast or DEFAULT_PARSE_AST
    )
    return unpack_feature(byte_arr, parse_ast)


def get_per_store_features_from_states(
    states: List[Union[State, StateObject]], task: "SearchTask",
    max_n_bufs: Optional[int] = None,
    parse_ast: Optional[bool] = None
) -> np.ndarray:
    """Get per-store features from measurement input/result pairs

    Parameters
    ----------
    states: List[Union[State, StateObject]]
        The input states

    Returns
    -------
    features: np.ndarray
        Feature vectors
    """
    if isinstance(states[0], State):
        state_objects = [s.state_object for s in states]
    elif isinstance(states[0], StateObject):
        state_objects = states
    byte_arr = _ffi_api.GetPerStoreFeaturesFromStates(
        state_objects, task, max_n_bufs or DEFAULT_MAX_N_BUFS,
        parse_ast or DEFAULT_PARSE_AST
    )
    return unpack_feature(byte_arr, parse_ast)[0]


def get_per_store_feature_names(max_n_bufs: Optional[int] = None, parse_ast: Optional[bool] = None) -> List[str]:
    """Get the name of every element in the feature vector. Use this for debug and inspection.

    Parameters
    ----------
    max_n_bufs: int
        The maximum number of extracted buffers for one statement
    parse_ast: bool
        Reture AST+features if set True

    Returns
    -------
    names: List[str]
        The names of elements in the flatten feature vector
    """
    return _ffi_api.GetPerStoreFeatureNames(max_n_bufs or DEFAULT_MAX_N_BUFS, parse_ast or DEFAULT_PARSE_AST)
