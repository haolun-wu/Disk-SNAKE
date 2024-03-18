import jax
import jax.numpy as jnp

from flax import linen as nn
import optax
from flax.training import checkpoints

import numpy as np
import tensorflow_datasets as tfds

from tqdm import tqdm
from functools import partial

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from typing import Any
from flax.training import train_state

import gin
import unet

gin.external_configurable(optax.adam)
gin.external_configurable(optax.sgd)

@gin.configurable
def get_datum_shape(shape=(32, 32)):
    return shape

@gin.configurable
def get_model(model_class, training=True):
    return model_class(training=training)

@gin.configurable
def get_optimizer(opt_class=optax.adam, learning_rate=1e-3):
    return opt_class(learning_rate)

@gin.configurable
def get_init_inputs(shapes=[(1, 32, 32)]):
    return [jnp.ones(shape, dtype=jnp.int8) for shape in shapes]
    
class TrainState(train_state.TrainState):
    batch_stats: Any

def create_train_state(learning_rate):
    """
    Creates initial `TrainState`
    
    This is a very common partern which working with flax
    (and other jax libs). The model is initialized using `init`
    and the TrainState object (a dataclass) is used to hold the 
    state of the whole model. This state can then easily be 
    saved and reloaded using flax.checkpoints
    """
    model = get_model()
    param_key, latent_key = jax.random.split(jax.random.PRNGKey(0))
    init_rngs = {'params': param_key, 'latents': latent_key}
    init_inputs = get_init_inputs()
    variables = model.init(init_rngs, *init_inputs)
    tx = get_optimizer()
    return TrainState.create(apply_fn=model.apply,
                             params=variables['params'],
                             batch_stats=variables['batch_stats'],
                             tx=tx)


def get_element_counts(x_batch, padding_masks=None):
    """
    count the number of (non-padded) elements. 
    returns:
      datum_size (the size of each datum including padded elements)
      datum_ndim (the number of non-padded elements in each element of the batch)
      datum_ax : the indices of the axes over which the data are (i.e. not the batch dim)
    """
    datum_shape = get_datum_shape()
    datum_dims = len(datum_shape)
    batch_shape = x_batch.shape[:-datum_dims]
    datum_size = int(np.prod(datum_shape))
    datum_ax = np.arange(datum_dims) + len(batch_shape)
    if padding_masks is not None:
        datum_ndim = jnp.sum(jnp.logical_not(padding_masks), axis=datum_ax).astype(jnp.float32)
    else:
        datum_ndim = datum_size
    return datum_size, datum_ndim, datum_ax


def mask_out_batch(key, x_batch, padding_masks=None):
    """
    Mask out the images (making way for 0 to represent a mask)
    Note that if we use padding_masks then the (additional) masks will not be in the padding.
    """
    datum_size, datum_ndim, datum_ax = get_element_counts(x_batch, padding_masks)
    batch_shape = x_batch.shape[:-len(get_datum_shape())]
    # generate the number of masks
    p_key, N_key, perm_key = jax.random.split(key, 3)
    p = jax.random.uniform(p_key, shape=batch_shape)
    N = tfd.Binomial(total_count=datum_ndim, probs=p).sample(seed=N_key)
    N_tilde = jnp.fmin(N+1, datum_ndim) # mask an additional element
    
    # randomly apply the masks
    masks = jnp.where(jnp.arange(datum_size) < N_tilde[..., None], True, False)
    if padding_masks is None:
        masks = jax.random.permutation(perm_key, masks, axis=-1, independent=True)
    else:
        flat_padding = padding_masks.reshape(*batch_shape, -1)
        u = jax.random.uniform(perm_key, shape=flat_padding.shape)
        u = jnp.where(flat_padding, 1, u)
        ind = jnp.argsort(u, axis=1)
        tiled_range = jnp.tile(jnp.arange(batch_shape[0])[:, None], [1, flat_padding.shape[1]])
        masks = masks[tiled_range, ind]
        
    masks = masks.reshape(*(batch_shape + get_datum_shape()))
    x_masked = jnp.where(masks, 0, x_batch + 1)
    return p, N, masks, x_masked

@jax.jit
def train_step(key, state, x_batch, padding_masks=None):
    """Perform a single training step.
    
    key: a jax.random.PRNGKey
    state: a flax training state (see create_train_state)
    x_batch: the data we're modelling. This is a numpy array of integers or shape:
        [batch_size,] + [datum_shape]
      where datum shape must match `get_datum_shape()`
    padding_masks: (optional) a numpy boolean array of the same shape as x_batch which tells us which elements of x_batch are just padding. 
    
    conventions:
        in the 'x_batch' array, padded elements take the value -1
        x_batch *can* use the integer 0, we shift everything by +1 during masking. 
        
    """
    datum_size, datum_ndim, datum_ax = get_element_counts(x_batch, padding_masks)
    
    def loss_fn(params):
        p_mask, N, masks, x_masked = mask_out_batch(key, x_batch, padding_masks)
        
        logits, new_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x_masked, #padding_masks,
            mutable=['batch_stats'])
        
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, x_batch) # batch_shape + datum_shape
        loss = jnp.sum(masks * loss, axis=datum_ax) / jnp.sum(masks, axis=datum_ax)  # mean over masked-out data
        
        p_empirical = N / datum_ndim
        factor = (1 - p_empirical) / (1 - p_mask)
        loss *= factor
        loss *= datum_ndim
        
        loss = jnp.mean(loss) # mean over batch
        return loss, (new_state, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_state, logits)), grads = grad_fn(state.params)

    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_state['batch_stats'])
    
    return new_state


def train_loop(key, state, num_steps, dataset):
    """Train for num_steps iterations"""

    for i in tqdm(range(num_steps)):
        key, newkey = jax.random.split(key)
        state = train_step(newkey, state, next(dataset))

    return state

@partial(jax.jit, static_argnames=['num_leaps', 'num_samples'])
def generate_samples(key, state, num_samples, num_leaps=1, padding_mask=None):
    """
    Generate a sample from the diffusion model byt runing the reverse diffusion process. 
    """
    datum_shape = get_datum_shape()
    if padding_mask is not None:
        datum_size = jnp.sum(jnp.logical_not(padding_mask))
    else:
        datum_size = np.prod(datum_shape)
    
    def reverse_diffusion_step(args, _):
        key, x = args

        # push forward the NN to get logits
        model = get_model(training=False)
        logits = model.apply({'params': state.params, 'batch_stats': state.batch_stats}, x)
        logits -= jax.nn.logsumexp(logits, axis=3, keepdims=True) # make log-probs for each dimension
        state_dim = logits.shape[-1]

        # select which pixels to flip
        x = x.reshape(num_samples, datum_size)
        key, newkey = jax.random.split(key)
        u = jax.random.uniform(newkey, (num_samples, datum_size))
        u = jnp.where(x==0, u, 0)
        if padding_mask is not None:
            u = jnp.where(padding_mask, 0, u)
        index = jnp.argsort(u, axis=1)[:, ::-1][:, :num_leaps]
        tiled_range = jnp.tile(jnp.arange(num_samples)[:, None], [1, num_leaps])

        # select which state to flip each chosen pixel into
        logits = jnp.rollaxis(logits, -1, 1)  # ensure correct ravelling of width & height (now batch x S x W x H)
        logits = logits.reshape(num_samples, state_dim, datum_size)
        logits = logits[tiled_range, :, index] # only select flipped pixels: logits are now (batch x leap x S).
        key, newkey = jax.random.split(key)
        new_states = jax.random.categorical(newkey, logits)  # batch x leap
        x = x.at[tiled_range, index].set(new_states + 1) # +1 reserves 0 for mask

        return (key, x.reshape((num_samples,) + datum_shape)), None

    x = jnp.zeros((num_samples,) + datum_shape, dtype=jnp.int8)
    loop_length = datum_size // num_leaps  # TODO deal with modulo
    (key, x), _ = jax.lax.scan(reverse_diffusion_step, init=(key, x), xs=None, length=loop_length)
            
    x =  x - 1  # stop reserving 0 for mask
    return x


@partial(jax.jit, static_argnames=['num_leap', 'num_particles'])
def smc_logprob(key, state, image, num_particles=32, num_leap=16):
    width, height = image.shape
    
    # initialize weights and z estimate
    log_z = 0.
    log_weights = jnp.zeros((num_particles,)) - jnp.log(num_particles)
    
    # initialize particles. +1 reserves 0 for the mask-state.
    x = jnp.tile(image + 1, [num_particles, 1, 1])
    
    def add_masks(key, x, num_mask):
        """randomly mask out num_mask (additional) pixels in every particle"""
        x = x.reshape(num_particles, width * height)
        masks = jnp.where(x==0, True, False)
        u = jax.random.uniform(key, shape=x.shape)
        index = jnp.argsort(jnp.where(masks, 0, u), axis=1)[:, ::-1][:, :num_mask]
        tiled_range = jnp.tile(jnp.arange(num_particles)[:, None], [1, num_mask])
        x = x.at[tiled_range, index].set(0)
        return x.reshape(num_particles, width, height), index
    
    
    def step(key, x, log_z, log_weights, num_leap):
        """
        One SMC step of length num_leap
        Note that the log-probability is missing a Binomial term:
            logp += gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
        which represents the probabilty of choosing k particles to unmask from the N masked ones. 
        But this term cancels out with those in the proposal distribution, so we ignore both. 
        """
        
        # adptively resample (this construction is XLA compilable!)
        def resample(key, x, log_weights):
            key, newkey = jax.random.split(key)
            x = jax.random.choice(newkey, x, shape=(num_particles,), replace=True, p=jnp.exp(log_weights))
            log_weights = jnp.zeros((num_particles,)) - jnp.log(num_particles)
            return key, x, log_weights
        def dont_resample(key, x, log_weights):
            return key, x, log_weights   
        n_eff= 1./jnp.sum(jnp.square(jnp.exp(log_weights)))
        key, x, log_weights = jax.lax.cond(n_eff < 0.25*num_particles, resample, dont_resample, key, x, log_weights)
            
        # add masks at random
        key, newkey = jax.random.split(key)
        x_next, mask_index = add_masks(newkey, x, num_leap)

        # get log prob of reconstruction (recently masked elements only)
        model = get_model(training=False)
        logits = model.apply({'params': state.params, 'batch_stats': state.batch_stats}, x_next)
        logp = -optax.softmax_cross_entropy_with_integer_labels(logits, x - 1) # particles x width x height
        tiled_range = jnp.tile(jnp.arange(num_particles)[:, None], [1, num_leap])
        logp = logp.reshape(num_particles, width*height)[tiled_range, mask_index]  # num_particles x leaps
        logp = jnp.sum(logp, axis=1) # sum logprobs over leaps
        
        # update weights, and store the normalization constant
        log_weights += logp
        log_z_i = jax.nn.logsumexp(log_weights)
        log_z += log_z_i
        log_weights -= log_z_i
        
        return key, x_next, log_z, log_weights
        
    step_ = lambda args, _ : (step(*args, num_leap), None)  # dummy function to cope with scan's requirements
    (key, x, log_z, log_weights), _ = jax.lax.scan(step_, init=(key, x, log_z, log_weights), xs=None, length=width*height // num_leap)
    key, x, log_z, log_weights = step(key, x, log_z, log_weights, num_leap=width * height % num_leap) # deal with modulo
    return log_z
