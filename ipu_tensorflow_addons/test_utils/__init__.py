# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# importing this extends TensorFlowTestCase with addidional asserts
import ipu_tensorflow_addons.test_utils.test_case_extensions

from ipu_tensorflow_addons.test_utils.report_helper import ReportHelper

from ipu_tensorflow_addons.test_utils.hardware_test_utils import get_ci_num_ipus
from ipu_tensorflow_addons.test_utils.hardware_test_utils import has_ci_ipus
from ipu_tensorflow_addons.test_utils.hardware_test_utils import add_hw_ci_connection_options
from ipu_tensorflow_addons.test_utils.hardware_test_utils import test_may_use_ipus_or_model
from ipu_tensorflow_addons.test_utils.hardware_test_utils import test_uses_ipus
from ipu_tensorflow_addons.test_utils.hardware_test_utils import skip_on_hw
from ipu_tensorflow_addons.test_utils.hardware_test_utils import skip_if_not_enough_ipus
