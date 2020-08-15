import collections
import tensorflow as tf
import tf_slim as slim

Clone = collections.namedtuple('Clone',
                               ['outputs',
                                'scope',
                                'device',
                               ])

DeployedModel = collections.namedtuple('DeployedModel',
                                       ['train_op',
                                        'summary_op',
                                        'total_loss',
                                        'clones',
                                       ])

_deployment_params = {'num_clones': 1,
                      'clone_on_cpu': False,
                      'replica_id': 0,
                      'num_replicas': 1,
                      'num_ps_tasks': 0,
                      'worker_job_name': 'localhost',
                      'ps_job_name': 'ps'}


def create_clones(config, model_fn, args=None, kwargs=None):

  clones = []
  args = args or []
  kwargs = kwargs or {}
  with slim.arg_scope([slim.model_variable, slim.variable],
                      device=config.variables_device()):
    for i in range(0, config.num_clones):
      with tf.name_scope(config.clone_scope(i)) as clone_scope:
        clone_device = config.clone_device(i)
        with tf.device(clone_device):
          with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(),
                                 reuse=True if i > 0 else None):
            outputs = model_fn(*args, **kwargs)
          clones.append(Clone(outputs, clone_scope, clone_device))
  return clones


def _gather_clone_loss(clone, num_clones, regularization_losses):

  sum_loss = None

  clone_loss = None
  regularization_loss = None

  with tf.device(clone.device):
    all_losses = []
    clone_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOSSES, clone.scope)
    if clone_losses:
      clone_loss = tf.add_n(clone_losses, name='clone_loss')
      if num_clones > 1:
        clone_loss = tf.compat.v1.div(clone_loss, 1.0 * num_clones,
                            name='scaled_clone_loss')
      all_losses.append(clone_loss)
    if regularization_losses:
      regularization_loss = tf.add_n(regularization_losses,
                                     name='regularization_loss')
      all_losses.append(regularization_loss)
    if all_losses:
      sum_loss = tf.add_n(all_losses)

  if clone_loss is not None:
    tf.summary.scalar('/'.join(filter(None,
                                      ['Losses', clone.scope, 'clone_loss'])),
                      clone_loss)
  if regularization_loss is not None:
    tf.summary.scalar('Losses/regularization_loss', regularization_loss)
  return sum_loss


def _optimize_clone(optimizer, clone, num_clones, regularization_losses,
                    **kwargs):

  sum_loss = _gather_clone_loss(clone, num_clones, regularization_losses)
  clone_grad = None
  if sum_loss is not None:
    with tf.device(clone.device):
      # gradients = tf.GradientTape().gradient(sum_loss,**kwargs)
      # clone_grad = optimizer.apply_gradients
      clone_grad = optimizer.compute_gradients(sum_loss, **kwargs)
  return sum_loss, clone_grad


def optimize_clones(clones, optimizer,
                    regularization_losses=None,
                    **kwargs):

  grads_and_vars = []
  clones_losses = []
  num_clones = len(clones)
  if regularization_losses is None:
    regularization_losses = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
  for clone in clones:
    with tf.name_scope(clone.scope):
      clone_loss, clone_grad = _optimize_clone(
          optimizer, clone, num_clones, regularization_losses, **kwargs)
      if clone_loss is not None:
        clones_losses.append(clone_loss)
        grads_and_vars.append(clone_grad)

      regularization_losses = None

  total_loss = tf.add_n(clones_losses, name='total_loss')

  grads_and_vars = _sum_clones_gradients(grads_and_vars)
  return total_loss, grads_and_vars


def deploy(config,
           model_fn,
           args=None,
           kwargs=None,
           optimizer=None,
           summarize_gradients=False):

  # Gather initial summaries.
  summaries = set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

  # Create Clones.
  clones = create_clones(config, model_fn, args, kwargs)
  first_clone = clones[0]

  update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, first_clone.scope)

  train_op = None
  total_loss = None
  with tf.device(config.optimizer_device()):
    if optimizer:
      # Place the global step on the device storing the variables.
      with tf.device(config.variables_device()):
        global_step = slim.get_or_create_global_step()

      # Compute the gradients for the clones.
      total_loss, clones_gradients = optimize_clones(clones, optimizer)

      if clones_gradients:
        if summarize_gradients:
          summaries |= set(_add_gradients_summaries(clones_gradients))

        # Create gradient updates.
        grad_updates = optimizer.apply_gradients(clones_gradients,
                                                 global_step=global_step)
        update_ops.append(grad_updates)

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
          train_op = tf.identity(total_loss, name='train_op')
    else:
      clones_losses = []
      regularization_losses = tf.compat.v1.get_collection(
          tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
      for clone in clones:
        with tf.name_scope(clone.scope):
          clone_loss = _gather_clone_loss(clone, len(clones),
                                          regularization_losses)
          if clone_loss is not None:
            clones_losses.append(clone_loss)
          regularization_losses = None
      if clones_losses:
        total_loss = tf.add_n(clones_losses, name='total_loss')


    summaries |= set(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES,
                                       first_clone.scope))

    if total_loss is not None:
      # Add total_loss to summary.
      summaries.add(tf.summary.scalar('total_loss', total_loss))

    if summaries:
      # Merge all summaries together.
      summary_op = tf.compat.v1.summary.merge(list(summaries), name='summary_op')
    else:
      summary_op = None

  return DeployedModel(train_op, summary_op, total_loss, clones)


def _sum_clones_gradients(clone_grads):

  sum_grads = []
  for grad_and_vars in zip(*clone_grads):
    grads = []
    var = grad_and_vars[0][1]
    for g, v in grad_and_vars:
      assert v == var
      if g is not None:
        grads.append(g)
    if grads:
      if len(grads) > 1:
        sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
      else:
        sum_grad = grads[0]
      sum_grads.append((sum_grad, var))
  return sum_grads


def _add_gradients_summaries(grads_and_vars):

  summaries = []
  for grad, var in grads_and_vars:
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        grad_values = grad.values
      else:
        grad_values = grad
      summaries.append(tf.summary.histogram(var.op.name + ':gradient',
                                            grad_values))
      summaries.append(tf.summary.histogram(var.op.name + ':gradient_norm',
                                            tf.compat.v1.global_norm([grad_values])))
    else:
      tf.logging.info('Var %s has no gradient', var.op.name)
  return summaries


class DeploymentConfig(object):

  def __init__(self,
               num_clones=1,
               clone_on_cpu=False,
               replica_id=0,
               num_replicas=1,
               num_ps_tasks=0,
               worker_job_name='localhost',
               ps_job_name='ps'):
    if num_replicas > 1:
      if num_ps_tasks < 1:
        raise ValueError('When using replicas num_ps_tasks must be positive')
    if num_replicas > 1 or num_ps_tasks > 0:
      if not worker_job_name:
        raise ValueError('Must specify worker_job_name when using replicas')
      if not ps_job_name:
        raise ValueError('Must specify ps_job_name when using parameter server')
    if replica_id >= num_replicas:
      raise ValueError('replica_id must be less than num_replicas')
    self._num_clones = num_clones
    self._clone_on_cpu = clone_on_cpu
    self._replica_id = replica_id
    self._num_replicas = num_replicas
    self._num_ps_tasks = num_ps_tasks
    self._ps_device = '/job:' + ps_job_name if num_ps_tasks > 0 else ''
    self._worker_device = '/job:' + worker_job_name if num_ps_tasks > 0 else ''

  @property
  def num_clones(self):
    return self._num_clones

  @property
  def clone_on_cpu(self):
    return self._clone_on_cpu

  @property
  def replica_id(self):
    return self._replica_id

  @property
  def num_replicas(self):
    return self._num_replicas

  @property
  def num_ps_tasks(self):
    return self._num_ps_tasks

  @property
  def ps_device(self):
    return self._ps_device

  @property
  def worker_device(self):
    return self._worker_device

  def caching_device(self):
    """Returns the device to use for caching variables.

    Variables are cached on the worker CPU when using replicas.

    Returns:
      A device string or None if the variables do not need to be cached.
    """
    if self._num_ps_tasks > 0:
      return lambda op: op.device
    else:
      return None

  def clone_device(self, clone_index):
    """Device used to create the clone and all the ops inside the clone.

    Args:
      clone_index: Int, representing the clone_index.

    Returns:
      A value suitable for `tf.device()`.

    Raises:
      ValueError: if `clone_index` is greater or equal to the number of clones".
    """
    if clone_index >= self._num_clones:
      raise ValueError('clone_index must be less than num_clones')
    device = ''
    if self._num_ps_tasks > 0:
      device += self._worker_device
    if self._clone_on_cpu:
      device += '/device:CPU:0'
    else:
      device += '/device:GPU:%d' % clone_index
    return device

  def clone_scope(self, clone_index):

    if clone_index >= self._num_clones:
      raise ValueError('clone_index must be less than num_clones')
    scope = ''
    if self._num_clones > 1:
      scope = 'clone_%d' % clone_index
    return scope

  def optimizer_device(self):
    """
    返回用于optimizer的设备
    """
    if self._num_ps_tasks > 0 or self._num_clones > 0:
      return self._worker_device + '/device:CPU:0'
    else:
      return ''

  def inputs_device(self):
    """
    返回用于输入的设备
    """
    device = ''
    if self._num_ps_tasks > 0:
      device += self._worker_device
    device += '/device:CPU:0'
    return device

  def variables_device(self):
    """
    返回用于在clone中创建的变量的设备。
    """
    device = ''
    if self._num_ps_tasks > 0:
      device += self._ps_device
    device += '/device:CPU:0'

    class _PSDeviceChooser(object):

      def __init__(self, device, tasks):
        self._device = device
        self._tasks = tasks
        self._task = 0

      def choose(self, op):
        if op.device:
          return op.device
        node_def = op if isinstance(op, tf.compat.v1.NodeDef) else op.node_def
        if node_def.op.startswith('Variable'):
          t = self._task
          self._task = (self._task + 1) % self._tasks
          d = '%s/task:%d' % (self._device, t)
          return d
        else:
          return op.device

    if not self._num_ps_tasks:
      return device
    else:
      chooser = _PSDeviceChooser(device, self._num_ps_tasks)
      return chooser.choose
