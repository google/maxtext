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
import max_utils
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.experimental.serialize_executable import deserialize_and_load


import pickle
import functools
from input_pipeline import input_pipeline_interface



def get_functional_train_with_signature(train_step, mesh, state_mesh_annotations, model, config):
  """ Get the shardings (both state and data) for train_step """
  functional_train = get_functional_train_step(train_step, model, config)
  functional_train.__name__ = "train_step"
  data_pspec = P(*config.data_sharding)
  state_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  data_sharding = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
  in_shardings = (state_mesh_shardings, data_sharding, None) # State, batch, rng
  out_shardings = (state_mesh_shardings, None) # State, metrics
  static_argnums = () # We partial out the static argnums of model and config
  donate_argnums = 0 # This is the index of the state - we allow the compiler to make use of this memory.
  return functional_train, in_shardings, out_shardings, static_argnums, donate_argnums

def get_functional_train_step(train_step, model, config):
  return functools.partial(train_step, model, config)

def get_functional_eval_with_signature(eval_step, mesh, state_mesh_annotations, model, config):
  """ Get the shardings (both state and data) for eval_step """
  functional_eval = get_functional_eval_step(eval_step, model, config)
  functional_eval.__name__ = "eval_step"
  data_pspec = P(*config.data_sharding)
  state_mesh_shardings = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), state_mesh_annotations)
  data_sharding = jax.tree_map(
      lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
  in_shardings = (state_mesh_shardings, data_sharding, None) # State, batch, rng
  out_shardings = None # metrics
  static_argnums = () # We partial out the static argnums of model, config
  donate_argnums = () # state will be kept instead of being donated in eval_step
  return functional_eval, in_shardings, out_shardings, static_argnums, donate_argnums

def get_functional_eval_step(eval_step, model, config):
  return functools.partial(eval_step, model, config)

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
  shaped_batch = input_pipeline_interface.get_shaped_batch(config)
  example_rng = jax.random.PRNGKey(0)
  shaped_input_args = (state, shaped_batch, example_rng)
  shaped_input_kwargs = {}
  in_tree, out_tree = get_train_input_output_trees(partial_train, shaped_input_args, shaped_input_kwargs)
  p_train_step = deserialize_and_load(serialized_compiled, in_tree, out_tree)
  return p_train_step

# https://arxiv.org/pdf/2204.02311.pdf Appendix B
def calculate_tflops_training_per_device(num_model_parameters, config, log=True):
  """ Calculate training TFLOP"""
  learnable_weight_tflops = 6 * num_model_parameters * config.max_target_length * config.per_device_batch_size \
                                   / 10**12
  noncasual_attention_flops = 12 * config.num_query_heads * config.num_decoder_layers * config.head_dim \
                      * config.max_target_length**2 * config.per_device_batch_size / 10**12
  causal_attention_tflops = noncasual_attention_flops / 2 # due to causality in attention
  total_tflops = learnable_weight_tflops + causal_attention_tflops

  if log:
    print('Per train step:\n',
          f'Total TFLOPs: {total_tflops:.2f} \n',
          f'split as {100 * learnable_weight_tflops/total_tflops:.2f}% learnable weight flops',
          f'and {100 * causal_attention_tflops/total_tflops:.2f}% attention flops')
  return total_tflops, learnable_weight_tflops, causal_attention_tflops

# https://arxiv.org/pdf/2204.02311.pdf Appendix B
def calculate_tflops_prefill(num_model_parameters, prefill_length, config, log=True):
  """ Calculate training TFLOP"""
  learnable_weight_tflops = 2 * num_model_parameters * prefill_length \
                                   / 10**12
  noncasual_attention_flops = 4 * config.num_query_heads * config.num_decoder_layers * config.head_dim \
                      * prefill_length**2 * config.per_device_batch_size / 10**12
  causal_attention_tflops = noncasual_attention_flops / 2 # due to causality in attention
  total_tflops = learnable_weight_tflops + causal_attention_tflops

  if log:
    print('Per prefill step: \n',
          f'\tTotal TFLOPs: {total_tflops:.2f} \n',
          f'\t\tLearnable weight TFLOPs: {learnable_weight_tflops} ',
          f'({100 * learnable_weight_tflops/total_tflops:.2f})% of Total\n',
          f'\t\tCausal attention TFLOPs: {causal_attention_tflops} ',
          f'({100 * causal_attention_tflops/total_tflops:.2f})% of Total')
  return total_tflops, learnable_weight_tflops, causal_attention_tflops


def calc_not_sharded_params(shard,
                      expected_per_device_num_param,
                      key,
                      pspec,
                      num_devices_sharded
                      ):
  """Returns the number of params that aren't sharded."""
  new_num_p = np.prod(shard.data.shape)

  if new_num_p == expected_per_device_num_param:
    print(f'Input is sharded over {num_devices_sharded} devices as expected!')
  else:
    print(f'Expected {expected_per_device_num_param} params but got {new_num_p}')
    print(f'Off by a factor of {new_num_p // expected_per_device_num_param}')
    print('key path', key)
    print('pspec', pspec)
    if jax.process_index() == 0:
      return new_num_p
  return 0

def is_sharded_correctly(params):
  """Checks whether most params are sharded accros sharding axis.

  Given state params, function checks if over 99% of params are
  sharded accross 'fsdp', 'fsdp_transpose','sequence', 'tensor' axes.
  """
  num_not_sharded_params = [0]
  total_num_params = max_utils.calculate_num_params_from_pytree(params)

  def check_state_shardings(key, pytree):
    """Checks whether given pytree is sharded accross the sharding axes and
    updates the number of not sharded params."""
    pspec = pytree.sharding.spec
    num_devices_sharded = 1
    for axis in ['fsdp', 'fsdp_transpose','sequence', 'tensor']:
      num_devices_sharded *= pytree.sharding.mesh.shape[axis]

    num_params = max_utils.calculate_num_params_from_pytree(pytree)
    expected_num_p = num_params // num_devices_sharded
    for shard in pytree.addressable_shards:
      num_not_sharded_params[0] += calc_not_sharded_params(
        shard, expected_num_p, key, pspec, num_devices_sharded) // jax.local_device_count()

  jax.tree_util.tree_map_with_path(
    check_state_shardings,
    params,
    )
  return num_not_sharded_params[0] * 100 / (total_num_params * 1e9) < 1
