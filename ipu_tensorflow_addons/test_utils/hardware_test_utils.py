# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
from tensorflow.python.ipu import utils


def get_ci_num_ipus():
  return int(os.getenv('TF_IPU_COUNT', 0))


def has_ci_ipus():
  return get_ci_num_ipus() > 0


def add_hw_ci_connection_options(opts):
  opts.device_connection.enable_remote_buffers = True
  opts.device_connection.type = utils.DeviceConnectionType.ON_DEMAND


def test_may_use_ipus_or_model(num_ipus, func=None):
  """Test decorator for indicating that a test can run on both HW and Poplar
  IPU Model.
  Args:
  * num_ipus: number of IPUs required by the test.
  * func: the test function.
  """
  return test_uses_ipus(num_ipus=num_ipus, allow_ipu_model=True, func=func)


def test_uses_ipus(num_ipus, allow_ipu_model=False, func=None):
  """Test decorator for indicating how many IPUs the test requires. Allows us
  to skip tests which require too many IPUs.

  Args:
  * num_ipus: number of IPUs required by the test.
  * allow_ipu_model: whether the test supports IPUModel so that it can be
    executed without hardware.
  * func: the test function.
  """
  def decorator(f):
    def decorated(self, *args, **kwargs):
      num_available_ipus = get_ci_num_ipus()
      if num_available_ipus < num_ipus and not allow_ipu_model:
        self.skipTest(f"Requested {num_ipus} IPUs, but only "
                      f"{num_available_ipus} are available.")
      if num_available_ipus >= num_ipus:
        assert not ("use_ipu_model" in os.getenv(
            'TF_POPLAR_FLAGS',
            "")), "Do not set use_ipu_model when running HW tests."
      return f(self, *args, **kwargs)

    return decorated

  if func is not None:
    return decorator(func)

  return decorator


def skip_on_hw(func):
  """Test decorator for skipping tests which should not be run on HW."""
  def decorator(f):
    def decorated(self, *args, **kwargs):
      if has_ci_ipus():
        self.skipTest("Skipping test on HW")

      return f(self, *args, **kwargs)

    return decorated

  return decorator(func)


def skip_if_not_enough_ipus(self, num_ipus):
  num_available_ipus = get_ci_num_ipus()
  if num_available_ipus < num_ipus:
    self.skipTest(f"Requested {num_ipus} IPUs, but only "
                  f"{num_available_ipus} are available.")
