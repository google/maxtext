"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=bare-except, consider-using-generator
"""Utils that are only interesting to MaxText. """

import jax
from jax.sharding import PartitionSpec as P
from jax.experimental.serialize_executable import deserialize_and_load


import maxtext_utils
import pickle
import functools
import input_pipeline
import optax



from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

def get_functional_train_with_signature(train_step, mesh, state_mesh_annotations, model, config):
  """ Get the shardings (both state and data) for train_step """
  functional_train = get_functional_train_step(train_step, model, config)
  data_pspec = P(*config.data_sharding)
  state_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  data_sharding = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
  in_shardings = (state_mesh_shardings, data_sharding, None) # State, batch, rng
  out_shardings = (state_mesh_shardings, None, None) # State, metrics, rng
  static_argnums = () # We partial out the static argnums of model and config
  donate_argnums = 0 # This is the index of the state - we allow the compiler to make use of this memory.
  return functional_train, in_shardings, out_shardings, static_argnums, donate_argnums

def get_functional_train_step(train_step, model, config):
  return functools.partial(train_step, model, config)

def get_optimizer(config, learning_rate_schedule):
  """ Create AdamW Optimizer following Llama2's training details, see https://arxiv.org/pdf/2307.09288.pdf section 2.2 """
  return optax.adamw(
    learning_rate_schedule,
    b1=config.adam_b1,
    b2=config.adam_b2,
    eps=config.adam_eps,
    eps_root=config.adam_eps_root,
    weight_decay=config.adam_weight_decay,
  )

def load_compiled(config, partial_train, state):
  """ # Loading a serialized compiled train step function."""
  # Currently partial_train and state  are needed to reconstruct
  # input/output shapes to construct the in_trees and out_trees for load API
  # Parker is working on a serializing these
  def load_serialized_compiled(save_name):
    with open(save_name, "rb") as f:
      serialized_compiled = pickle.load(f)
    return serialized_compiled

  def get_train_input_output_trees(func, input_args, input_kwargs):
    _, in_tree_recreated = jax.tree_util.tree_flatten((input_args, input_kwargs))
    out_shaped = jax.eval_shape(func, *input_args, **input_kwargs)
    _, out_tree_recreated = jax.tree_util.tree_flatten(out_shaped)
    return in_tree_recreated, out_tree_recreated

  serialized_compiled = load_serialized_compiled(config.compiled_trainstep_file)
  shaped_batch = input_pipeline.get_shaped_batch(config)
  example_rng = jax.random.PRNGKey(0)
  shaped_input_args = (state, shaped_batch, example_rng)
  shaped_input_kwargs = {}
  in_tree, out_tree = get_train_input_output_trees(partial_train, shaped_input_args, shaped_input_kwargs)
  p_train_step = deserialize_and_load(serialized_compiled, in_tree, out_tree)
  return p_train_step

def get_abstract_state(model, tx, config, rng, mesh):
  """ Get a shaped abstraction of the state (including optimizer)"""
  init_train_state_partial = functools.partial(max_utils.init_train_state, model, tx,
                                              config)
  abstract_state = jax.eval_shape(init_train_state_partial, rng)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  unboxed_abstract_state = max_utils.unbox_logicallypartioned_trainstate(abstract_state)

  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return unboxed_abstract_state, state_mesh_annotations
