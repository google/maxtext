#! /usr/bin/python3
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

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more
import jax
import os

jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS","") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
XLA_DUMP_DIR="/tmp/xla_dumps"
os.environ["XLA_FLAGS"]=f"--xla_dump_to={XLA_DUMP_DIR}"
print(f"Found {jax.device_count()} devices.")

from typing import Sequence
import datetime
from absl import app
from flax.linen import partitioning as nn_partitioning
import numpy as np
import optax
from tensorboardX import SummaryWriter

from layers import Transformer
import pyconfig
import tensorflow_datasets as tfds
from input_pipeline import get_datasets
from input_pipeline import preprocess_dataset
import max_utils
import ucb
import checkpointing

import jax.numpy as jnp
from jax import random
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

from jax.experimental.compilation_cache import compilation_cache as cc

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

import max_logging
cc.initialize_cache(os.path.expanduser("~/jax_cache"))

# https://arxiv.org/pdf/2204.02311.pdf Appendix B
def calculate_training_tflops(num_model_parameters, config):
  learnable_weight_tflops = 6 * num_model_parameters * config.max_target_length * config.per_device_batch_size \
                                   / 10**12
  attention_tflops = 12 * config.num_heads * config.num_decoder_layers * config.head_dim * config.max_target_length**2 \
                     * config.per_device_batch_size / 10**12
  total_tflops = learnable_weight_tflops + attention_tflops
  print(f'Per train step, total TFLOPs will be {total_tflops:.2f},',
        f'split as {100 * learnable_weight_tflops/total_tflops:.2f}% learnable weight flops',
        f'and {100 * attention_tflops/total_tflops:.2f}% attention flops')
  return total_tflops

def get_first_step(state):
  with jax.spmd_mode('allow_all'):
    return int(state.step)


def load_next_batch(train_iter, example_batch, config):
  """Loads the next batch. Can keep reusing the same batch for performance reasons """

  if config.reuse_example_batch and example_batch is not None:
    return example_batch
  else:
    return train_iter()

def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr):
  """Records scalar metrics to be written to tensorboard"""
  metrics['scalar'].update({
      'perf/step_time_seconds': step_time_delta.total_seconds()
  })
  metrics['scalar'].update({
      'perf/per_device_tflops' : per_device_tflops
  })
  metrics['scalar'].update({
      'perf/per_device_tflops_per_sec':
          per_device_tflops /
          step_time_delta.total_seconds()
  })
  metrics['scalar'].update({'learning/current_learning_rate': lr })


def write_metrics(writer, metrics, step, config):
  """Writes metrics to tensorboard"""
  with jax.spmd_mode('allow_all'):
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar",[]):
        writer.add_scalar(metric_name, metrics["scalar"][metric_name], step)
      for metric_name in metrics.get("scalars",[]):
        writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    full_log = step % config.log_period == 0

    max_logging.log(f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
          f"TFLOP/s: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
          f"loss: {metrics['scalar']['learning/loss']:.3f}")

    if full_log:
      max_logging.log(
          f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'"
      )
      writer.flush()

def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  return total_parameters

# -----------------------------------------------------------------------------
# Top-level Functions
# -----------------------------------------------------------------------------

def record_activation_metrics(output_metrics, intermediate_outputs, config):
  """ Adds the activation metrics to the metrics dict"""

  if config.scan_layers:
    metrics_dict = intermediate_outputs['intermediates']['decoder']['decoder']

    for layer_num in range(config.num_decoder_layers):
      output_metrics['scalar'][f'activ_fraction_zero/layer_{layer_num:03d}'] = \
        metrics_dict["activation_fraction_zero"][0][layer_num]
      output_metrics['scalar'][f'activ_mean/layer_{layer_num:03d}'] = metrics_dict["activation_mean"][0][layer_num]
      output_metrics['scalar'][f'activ_stdev/layer_{layer_num:03d}'] = metrics_dict["activation_stdev"][0][layer_num]
  else:
    for layer_num in range(config.num_decoder_layers):
      layer = intermediate_outputs['intermediates']['decoder'][f'layers_{layer_num}']
      output_metrics['scalar'][f'activ_fraction_zero/layer_{layer_num:03d}'] = layer["activation_fraction_zero"][0]
      output_metrics['scalar'][f'activ_mean/layer_{layer_num:03d}'] = layer["activation_mean"][0]
      output_metrics['scalar'][f'activ_stdev/layer_{layer_num:03d}'] = layer["activation_stdev"][0]

def train_step(model, config, state, grad_stats, data, dropout_rng):
  """

  Args:
    model: A nn.Module
    state: A pytree of the current state of the model
    data: Batch of data to apply to the model
    dropout_rng: A key to use to generate rng for dropout

  Returns:
    new_state: Same format as state.
    metrics: Dictionary of model metrics such as loss, training rate, etc.
    rng2: A new rng key that can be used in future calls.

  """
  # inputs, targets, segments, positions = apply_args
  rng1, gen_aqt_rng = jax.random.split(dropout_rng)
  aqt_rng, rng2 = jax.random.split(gen_aqt_rng)

  def loss_fn(params):
    logits, intermediate_outputs = model.apply({'params': params},
                         data['inputs'],
                         data['targets'],
                         data['inputs_segmentation'],
                         data['inputs_position'],
                         enable_dropout=config.enable_dropout,
                         rngs={'dropout': rng1, 'aqt': aqt_rng}, mutable='intermediates')
    # TODO: is optax xent as good as custom T5X one?
    xent = optax.softmax_cross_entropy_with_integer_labels(logits, data['targets'])
    # Mask out paddings at the end of each example.
    mask = data['inputs_segmentation'] != 0
    # Old code:
    xent = jnp.sum(xent * mask) / jnp.size(mask)
    # Maybe better? More unstable maybe.
    # xent = jnp.sum(xent * mask) / jnp.sum(mask)
    # TODO: mask out the prompt if training prefix-LM
    return xent, (intermediate_outputs, logits)

  scalar_metrics = {}

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (intermediate_outputs, logits)), grads = grad_fn(state.params)
  raw_grads = grads
  # clipping
  clip_global = config.clip_by_global_norm > 0.0
  clip_block_rms = config.clip_by_block_rms > 0.0
  clip_ucb = config.clip_by_ucb > 0.0
  assert clip_global + clip_block_rms + clip_ucb <= 1

  new_grad_stats, clip_grads, ucb_metrics = ucb.ucb_update(grad_stats, grads)
  ucb_metrics = max_utils.leaves_metrics('ucb', ucb_metrics)
  scalar_metrics.update(ucb_metrics)

  if clip_global:
    grads, _ = optax.clip_by_global_norm(config.clip_by_global_norm).update(grads, None, None)
  elif clip_block_rms:
    grads, _ = optax.clip_by_block_rms(config.clip_by_block_rms).update(grads, None, None)
  elif clip_ucb:
    grads = clip_grads
    #   adam = state.opt_state[0]
    #   def maybe_zero(t):
    #     mask = jnp.full(t.shape, is_spike, dtype = jnp.bool_)
    #     return jax.lax.select(mask, jnp.zeros_like(t), t)
    #   new_adam = optax.ScaleByAdamState(
    #     mu=jax.tree_map(maybe_zero, adam.mu),
    #     nu=jax.tree_map(maybe_zero, adam.nu),
    #     count=jax.tree_map(maybe_zero, adam.count),
    #   )
    #   opt = new_adam, state.opt_state[1]
  else:
    pass # no clipping

  opt = state.opt_state

  updates, new_opt_state = state.tx.update(grads, opt, state.params)
  new_params = optax.apply_updates(state.params, updates)
  new_state = state.replace(
      step=state.step + 1,
      params=new_params,
      opt_state=new_opt_state,
  )

  scalar_metrics.update({
      'token_count': jnp.sum(data['inputs_segmentation'] != 0),
      'per_example_min_token_count': jnp.min(jnp.sum(data['inputs_segmentation'] != 0, axis=1), axis=0),
      'learning/loss': loss,
      'learning/logits_norm' : max_utils.l2norm_pytree(logits),
      'learning/grad_norm' : max_utils.l2norm_pytree(grads),
      'learning/raw_grad_norm' : max_utils.l2norm_pytree(raw_grads),
      'learning/weight_update_norm': max_utils.l2norm_pytree(updates),
      'learning/adam_mu_norm' : max_utils.l2norm_pytree(opt[0].mu),
      'learning/adam_nu_norm' : max_utils.l2norm_pytree(opt[0].nu),
      'learning/adam_count' : opt[0].count,
      'learning/param_norm' : max_utils.l2norm_pytree(new_state.params)
  })
  trees = {
    'logits': logits,
    'grads': grads,
    'raw_grads': raw_grads,
    'updates': updates,
    'mu': opt[0].mu,
    'nu': opt[0].nu,
    'params': new_state.params,
  }
  trees = max_utils.rms_leaves(trees)
  # Add means.
  trees_means = {k: max_utils.mean_pytree(v) for k,v in trees.items() }
  trees_metrics = max_utils.leaves_metrics('rms', trees)
  trees_means_metrics = max_utils.leaves_metrics('rms_means', trees_means)
  scalar_metrics.update(trees_metrics)
  scalar_metrics.update(trees_means_metrics)

  # for k in sorted(scalar_metrics.keys()):
  #   print(k)
  # assert False

  metrics = {
    'scalar': scalar_metrics,
    'scalars': {},
  }
  if config.record_internal_nn_metrics:
    record_activation_metrics(metrics, intermediate_outputs, config)

  return new_state, new_grad_stats, metrics, rng2


def train_loop(config, state=None):
  """Main Training loop.

  Args:
    config:
    state:
    ckpt_path:

  Returns:

  """
  writer = SummaryWriter(config.tensorboard_dir)
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(config.checkpoint_dir,
                                                                     config.enable_checkpointing,
                                                                     config.async_checkpointing)
  # Initial PRNG Keys
  init_rng, nextrng = random.split(random.PRNGKey(config.init_weights_seed), 2)

  # Model and Optimizer definition
  model = Transformer(config)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(
      learning_rate=config.learning_rate, total_steps=config.learning_rate_schedule_steps
  )

  # We use AdamW following Llama2's training details, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
  adam = optax.adamw(
      max_utils.create_learning_rate_schedule(
          learning_rate=config.learning_rate, total_steps=config.learning_rate_schedule_steps
      ),
      b1=config.adam_b1,
      b2=config.adam_b2,
      eps=config.adam_eps,
      eps_root=config.adam_eps_root,
      weight_decay=config.adam_weight_decay,
  )

  # tx = optax.chain(
  #     optax.chain(
  #       optax.identity(),
  #       optax.clip_by_global_norm(config.clip_by_global_norm),
  #       optax.clip_by_block_rms(config.clip_by_block_rms),
  #     ),
  #     adam,
  # )

  tx = adam

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  state, state_mesh_annotations = max_utils.setup_initial_state(model, tx, config, init_rng, mesh, checkpoint_manager)

  # import json
  # def sz(t):
  #   from functools import reduce # Valid in Python 2.6+, required in Python 3
  #   import operator
  #   return reduce(operator.mul, t.shape, 1)

  # print(json.dumps(jax.tree_map(jnp.shape, state.params), sort_keys=True, indent=4))
  # print(json.dumps(jax.tree_map(sz, state.params), sort_keys=True, indent=4))

  # Set up datasets.
  read_config = tfds.ReadConfig(
    shuffle_seed = config.data_shuffle_seed,
  )
  train_ds, eval_ds = get_datasets(
      config=config,
      read_config = read_config,
  )
  train_iter, _, _, _ = preprocess_dataset(
    config,
    mesh,
    train_ds, eval_ds,
    vocab_path=os.path.join(config.base_output_directory, config.vocab_relative_path),
    data_shuffle_seed = config.data_shuffle_seed,
  )


  grad_stats = ucb.ucb_init(state.params)
  data_pspec = P(*config.data_sharding)

  num_model_parameters = calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/10**9:.3f} billion")
  per_device_tflops = calculate_training_tflops(num_model_parameters, config)

  # Define compiled top-level functions.
  p_train_step = pjit(
    train_step,
    in_shardings=(state_mesh_annotations, None, data_pspec, None),
    out_shardings=(state_mesh_annotations, None, None, None),
    static_argnums=(0,1,),
    donate_argnums=2)

  example_batch = None
  last_step_completion = datetime.datetime.now()

  local_metrics_file = open(config.metrics_file, 'a', encoding="utf8") if config.metrics_file else None
  running_gcs_metrics = [] if config.gcs_metrics else None

  # for i in range(3001):
  #   example_batch = load_next_batch(train_iter, example_batch, config)
  #   if (i%100==0): max_logging.log(f"forwarding iterator {i}")
  # assert False

  first_step = get_first_step(state)
  for i in range(first_step):
    example_batch = load_next_batch(train_iter, example_batch, config)
    max_logging.log(f"forwarding iterator {i}")

  for step in np.arange(get_first_step(state), config.steps):
    example_batch = load_next_batch(train_iter, example_batch, config)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state, grad_stats, metrics, nextrng = p_train_step(
          model, config, state, grad_stats, example_batch, nextrng
      )

    new_time = datetime.datetime.now()
    record_scalar_metrics(metrics, new_time - last_step_completion,  per_device_tflops, learning_rate_schedule(step))
    write_metrics(writer, metrics, step, config)
    last_step_completion = new_time

    if step > 0 and step % config.save_period == 0 and checkpoint_manager is not None:
      checkpoint_manager.save(step, state)
      max_logging.log("saved a checkpoint")

    if config.metrics_file:
      max_utils.write_metrics_locally(metrics, step, config, local_metrics_file)

    if config.gcs_metrics and jax.process_index() == 0:
      running_gcs_metrics = max_utils.write_metrics_for_gcs(metrics, step, config, running_gcs_metrics)

    # Start profiling at end of first step to avoid compilation.
    # Move before for loop to include.
    if step == 0:
      gcs_xla_destination = os.path.join(config.base_output_directory, config.run_name)
      max_utils.move_local_dir_to_gcs(XLA_DUMP_DIR, gcs_xla_destination)

    if step == 4:
      max_utils.activate_profiler(config)
    if step == 6:
      max_utils.deactivate_profiler(config)

  writer.close()

  return state

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
  debug_config = debug_configuration.DebugConfig(
    stack_trace_config = stack_trace_configuration.StackTraceConfig(
      collect_stack_trace = pyconfig.config.collect_stack_trace,
      stack_trace_to_cloud = pyconfig.config.stack_trace_to_cloud,
      stack_trace_interval_seconds = pyconfig.config.stack_trace_interval_seconds))
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  with diagnostic.diagnose(diagnostic_config):
    train_loop(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
