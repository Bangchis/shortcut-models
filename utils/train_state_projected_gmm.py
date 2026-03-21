###############################
#
#  Training state for projected diagonal GMM mode.
#  Extends TrainStateEma to support nested params:
#    params = {"model": model_params, "prior": prior_params}
#
###############################

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util
import optax
import functools
from typing import Any, Callable

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)

class TrainStateProjectedGMMEma(flax.struct.PyTreeNode):
    rng: Any
    step: int
    apply_fn: Callable = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any          # {"model": ..., "prior": ...}
    params_ema: Any      # {"model": ..., "prior": ...}
    tx: Any = nonpytree_field()
    opt_state: Any
    prior_grad_accum: Any
    prior_accum_count: Any

    @classmethod
    def create(cls, model_def, params, rng, tx=None, opt_state=None, **kwargs):
        if tx is not None and opt_state is None:
            opt_state = tx.init(params)
        prior_grad_accum = jax.tree_map(jnp.zeros_like, params["prior"])
        prior_accum_count = jnp.array(0, dtype=jnp.int32)

        return cls(
            rng=rng, step=1, apply_fn=model_def.apply, model_def=model_def,
            params=params, params_ema=params,
            tx=tx, opt_state=opt_state,
            prior_grad_accum=prior_grad_accum,
            prior_accum_count=prior_accum_count,
            **kwargs,
        )

    # Call model with given or default params.
    def __call__(self, *args, params=None, method=None, **kwargs):
        if params is None:
            model_params = self.params["model"]
        elif isinstance(params, dict) and "model" in params:
            model_params = params["model"]
        else:
            model_params = params
        variables = {"params": model_params}
        if isinstance(method, str):
            method = getattr(self.model_def, method)
        return self.apply_fn(variables, *args, method=method, **kwargs)

    def call_model(self, *args, params=None, method=None, **kwargs):
        return self.__call__(*args, params=params, method=method, **kwargs)

    def call_model_ema(self, *args, params=None, method=None, **kwargs):
        return self.__call__(*args, params={"model": self.params_ema["model"]}, method=method, **kwargs)

    def get_prior_params(self, use_ema=False):
        if use_ema:
            return self.params_ema["prior"]
        return self.params["prior"]

    # Tau should be close to 1, e.g. 0.999.
    def update_ema(self, tau):
        new_model_ema = jax.tree_map(
            lambda p, tp: p * (1-tau) + tp * tau,
            self.params["model"],
            self.params_ema["model"],
        )
        # Keep prior EMA equal to online prior (no EMA-GMM behavior).
        new_params_ema = {
            "model": new_model_ema,
            "prior": self.params["prior"],
        }
        return self.replace(params_ema=new_params_ema)

    # For pickling.
    def save(self):
        return {
            'params': self.params,
            'params_ema': self.params_ema,
            'opt_state': self.opt_state,
            'step': self.step,
            'prior_grad_accum': self.prior_grad_accum,
            'prior_accum_count': self.prior_accum_count,
        }

    def load(self, data):
        return self.replace(**data)
