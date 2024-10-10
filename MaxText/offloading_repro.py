# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import jax
import jax.sharding
from jax.experimental.compute_on import compute_on
import jax.numpy as jnp
import numpy as np

import argparse
from flax import linen as nn
from jax.sharding import PartitionSpec as P
from flax.training import train_state
import optax

def with_memory_kind(t, memory_kind):
  return jax.tree_util.tree_map(
      lambda x: x.with_memory_kind(kind=memory_kind), t
  )

dtype = jnp.float32

class DummyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # A simple linear layer
        return nn.Dense(features=12376)(x)
    
def get_abstract_state(model, tx, mesh, rng):
    init_state_partial = functools.partial(init_intial_state, model, tx)
    abstract_state = jax.eval_shape(init_state_partial, rng)
    state_logical_annotations = nn.get_partition_spec(abstract_state)
    state_mesh_shardings = nn.logical_to_mesh_sharding(
        state_logical_annotations, mesh
    )
    return state_mesh_shardings

def init_intial_state(model, tx, rng):
    dummy_input = jnp.arange(0, 12376, dtype = dtype)
    # Initialize parameters
    params = model.init(rng, dummy_input)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return state

def create_train_state(model, mesh, rng):
    # Instantiate the model
    learning_rate = 0.001
    weight_decay = 1e-4
    
    # Define an optimizer
    tx = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    state_mesh_shardings = get_abstract_state(model, tx, mesh, rng)
    # Create the TrainState
    init_state_partial = functools.partial(init_intial_state, model, tx)
    state = jax.jit(
        init_state_partial,
        in_shardings=None,
        out_shardings=state_mesh_shardings,
    )(rng)
    return state, state_mesh_shardings

def train_step(model, state, batch):
    def loss_fn(model, batch, params):
        predictions = model.apply(
            params,
            batch
        )
        return jnp.mean((predictions - batch))
    grad_func = jax.value_and_grad(loss_fn, argnums=2)
    _, grad = grad_func(model, batch, state.params)
    new_state = state.apply_gradients(grads=grad)
    return new_state

# train_loop is called from main

def train_loop(output_path):
    # setting  up model and mesh
    rng = jax.random.PRNGKey(0)
    model = DummyModel()
    
    grid = np.array(jax.devices()).reshape((2, 2))
    mesh = jax.sharding.Mesh(grid, ('data', 'model'))

    # creating train_state
    state, state_mesh_shardings = create_train_state(model, mesh, rng)
    
    
    data_pspec = P(('data','model',))
    data_sharding = jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
    inputs = jnp.ones((12376))
    inputs = jax.device_put(inputs, with_memory_kind(data_sharding, 'device'))
    p_train_step = jax.jit(train_step,
            in_shardings=(state_mesh_shardings, data_sharding),
            out_shardings=state_mesh_shardings,
            donate_argnums=(1,),
            static_argnums=(0,))


    for step in range(10):
        if step == 5:
            jax.profiler.start_trace(output_path)
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            state = p_train_step(model, state, inputs)

    jax.profiler.stop_trace()
    print(f"Profile saved at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    test_args, _ = parser.parse_known_args()

    train_loop(test_args.output_path)
