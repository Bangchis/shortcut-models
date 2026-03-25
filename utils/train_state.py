###############################
#
#  Structures for managing training of flax networks.
#
###############################

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util
import optax
import functools
from typing import Any, Callable, Optional

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

# Contains model params and optimizer state.
class TrainStateEma(flax.struct.PyTreeNode):
    rng: Any
    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    source_apply_fn: Optional[Callable] = nonpytree_field(default=None)
    source_model_def: Any = nonpytree_field(default=None)
    params: Any
    params_ema: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(
        cls,
        model_def,
        params,
        rng,
        tx=None,
        opt_state=None,
        source_model_def=None,
        source_params=None,
        **kwargs,
    ):
        if tx is not None and opt_state is None:
            init_params = params if source_model_def is None else {
                'backbone': params,
                'source': source_params,
            }
            opt_state = tx.init(init_params)

        if source_model_def is not None:
            params = {
                'backbone': params,
                'source': source_params,
            }
            source_apply_fn = source_model_def.apply
        else:
            source_apply_fn = None

        return cls(
            rng=rng,
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            source_apply_fn=source_apply_fn,
            source_model_def=source_model_def,
            params=params,
            params_ema=params,
            tx=tx, opt_state=opt_state, **kwargs,
        )

    # Call model_def.apply_fn.
    def __call__(self, *args, params=None, method=None, **kwargs,):
        if params is None:
            params = self.params
        variables = {"params": params}
        if isinstance(method, str):
            method = getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method, **kwargs)

    def call_model(self, *args, params=None, method=None, **kwargs):
        if params is None:
            params = self.params
        if self.source_apply_fn is not None:
            params = params['backbone']
        return self.__call__(*args, params=params, method=method, **kwargs)
    
    def call_model_ema(self, *args, params=None, method=None, **kwargs):
        if params is None:
            params = self.params_ema
        if self.source_apply_fn is not None:
            params = params['backbone']
        return self.__call__(*args, params=params, method=method, **kwargs)

    def call_source(self, *args, params=None, method=None, **kwargs):
        if self.source_apply_fn is None:
            raise ValueError("Source model is not initialized for this train state.")
        if params is None:
            params = self.params
        if 'source' in params:
            params = params['source']
        variables = {"params": params}
        if isinstance(method, str):
            method = getattr(self.source_model_def, method)
        return self.source_apply_fn(variables, *args, method=method, **kwargs)

    def call_source_ema(self, *args, params=None, method=None, **kwargs):
        if self.source_apply_fn is None:
            raise ValueError("Source model is not initialized for this train state.")
        if params is None:
            params = self.params_ema
        if 'source' in params:
            params = params['source']
        variables = {"params": params}
        if isinstance(method, str):
            method = getattr(self.source_model_def, method)
        return self.source_apply_fn(variables, *args, method=method, **kwargs)

    # Tau should be close to 1, e.g. 0.999.
    def update_ema(self, tau):
        new_params_ema = jax.tree_map(
            lambda p, tp: p * (1-tau) + tp * tau, self.params, self.params_ema
        )
        return self.replace(params_ema=new_params_ema)

    # For pickling.
    def save(self):
        return {
            'params': self.params,
            'params_ema': self.params_ema,
            'opt_state': self.opt_state,
            'step': self.step,
        }
    
    def load(self, data):
        return self.replace(**data)
