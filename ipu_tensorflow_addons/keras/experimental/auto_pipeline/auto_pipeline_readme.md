# Auto Pipeline Readme
## What is a valid pipeline?
A pipeline stage assignment can be most easily described as $n_{ipu}$ ranges $[a_i,b_i]$ where $b_{i}+1=a_{i+1}$
## What is a good pipeline?
We should aim to find a balanced pipeline stage assignment. Why?

- A model fails to run on two IPUs because the second IPU encounters an OOM error. However, it may fit if we move a few layers from the second IPU to the first IPU.
- A pipeline is bottlenecked by the slowest IPU. 

We need to find a pipeline stage assignment such that every IPU has enough memory for the model's layers, with cycles not differing too much among all IPUs. In the next few sections, we start by looking at the simplest case, when we only want to balance cycles between IPUs. We then expand on the algorithm to consider other factors, such as the memory constraint and the transfer cycles.
## How to find a pipeline?
Assume for now we know we can estimate, for a range of layers(to run on a single IPU) the following quantities:

- The memory requirement of the layers, $memory[a,b]$.
- The cycle count required to complete the computation, $cycles[a,b]$.

How can we find a pipeline stage assignment? Well, it turns out to be a simple dynamic programming problem. 

### Split based on cycle number
Letting $cost[a,b]=cycles[a,b]$, we want to minimize the maximum cycle count across all IPUs.
```python
minimize max(cost[a_i,b_i] for i in range(n_ipu))
```
We can solve this problem using dynamic programming. Let `partition_cost[now_ipu,now_end]` be the overall cost to partition `[0,now_end]` layers on `[0,now_ipu]` pipeline stages. Then 
```python
partition_cost[now_ipu,now_end]=min(
  max(
    # Maximum cycle for previous now_ipu IPUs
    partition_cost[now_ipu-1][last_end],

    # Computation cycle
    cost[last_end+1, now_end]
  )
  for last_end in range(0, now_end)
)
partition_cost[now_ipu,i] = cost[0,i]
```

To recover the optimal pipeline plan via dynamic programming, we use another array `partition_from`, to indicate from which state `partition_cost[now_ipu,now_end]` is updated.

### How to make sure the generated pipeline stage assignment does not go OOM?
Simply, let $cost[a,b]=INF$ if the estimated memory consumption of running $[a,b]$ layer is above 894MB. 

If there exists a pipeline stage assignment such that every stage fits on one IPU, the cost of this pipeline stage assignment must be below $INF$. As the dynamic programming problem looks for a pipeline stage assignment with minimal cost, The algorithm will not return a `INF` cost partition.

### How to check the "variable in the same stage" constraint
If two layers both use the same `tf.Variable`, in the current implementation, the two layers need to be in the same pipeline stage.

Similarly, we use the `INF` value trick. For every range $[a,b]$ and for every layer $l$ in the range, check that every invocation of the layer $l$ is in the range $[a,b]$. If not, set $cost[a,b]=INF$. 

### Considering data transfer time
The above problem specification does not consider any data transfer cycles. A pipeline plan with perfectly balanced computation cycles may need to transfer more data between IPU. And thus such a plan may perform worse than a plan with less balanced cycles but fewer transferring cycles. 

We now want to minimize this, because in the current implementation of pipeline parallelism, each IPU starts transferring data only after the last IPU finishes computing. 
```python
  max(cost[a_i,b_i] for i in range(n_ipu)) 
+ max(transfer_cost[i, a_i, b_i] for i in range(n_ipu))
```
Where `transfer_cost[i,a_i,b_i]` is the cycle count for the `i`-th IPU to receive tensors from the CPU or the previous IPU and the cycle count to send tensors to the next IPU or CPU, if `[a_i,b_i]` layers were to run on this IPU.

Consider a more restricted form of the problem. What is the best partition, given that every IPU can spend no more than `max_transfer_cycle` cycles? We can solve this problem by changing very few things in the DP formula by similarly using the `INF` value trick. 
```python
partition_cost[now_ipu,now_end, max_transfer_cycle]=min(
  max(
    # Maximum cycle for previous now_ipu IPUs
    partition_cost[now_ipu-1][last_end],

    # Computation cycle
    INF if transfer_cost[last_end+1, now_end] > max_transfer_cycle
    else cost[last_end+1, now_end]
  )
  for last_end in range(0, now_end)
)

partition_cost[now_ipu, i, max_transfer_cycle] = 
  INF if transfer_cost[0, i] > max_transfer_cycle
  else cost[0,i]
```
Then the overall cost, including computation cycles and transferring cycles, is the optimal value from dynamic programming plus `max_transfer_cycle`. Among all choices of `max_transfer_cycle`, we can find one that minimizes the overall cost. 

Then what should be the choices of `max_transfer_cycle`? `max_transfer_cycle` can be as large as $10^7$ cycles. So it is impractical to check every value. We notice in an optimal pipeline plan, the value `max_transfer_cycle` must be in the `transfer_cost` array. In theory, this array can have $O(n_{ipu}*n_{layer}^2)$ different values. But for models with repeating substructures like BERT, the choice of `max_transfer_cycle` is usually $O(n_{layer})$.

The time complexity of the dynamic programming solution is $O(n_{ipu} n_{layer}^3)$. 

## How to estimate
### Per layer compilation/execution
The simplest way to find how much memory and how many cycles are required for a range is to compile the range with poplar. However, this is impractical because Poplar compilation is usually slow (for the large amount of optimization it is doing).

Instead, we perform a per-layer compilation and profiling. Then for each range, we estimate the memory and cycles from the per-layer profiles. 

### Cycle
The Poplar compiler can provide cycle estimation for every step in a compiled program. All we need to do for range computation estimation is to sum the estimated cycle count from each step. 

### Always alive memory
Compared to cycle estimation, memory estimation is more complicated. There are different categories of memory. Usually, temporary memory, vertex instance state and vertex code use the greatest proportion of memory.

Poplar can share some code programs. If two layers both perform a matrix multiplication of the same shape, their corresponding vertex code and control code are shared. But the corresponding vertex instance states are independent.

In the estimation, we implement layer-level sharing. If there are two layers from the same class and with the same configuration in a range, then some of the memory is shared. This is weaker than the sharing Poplar does, but it is likely to be enough for most models.

### Not always alive memory
During the computation of a layer, each IPU may use some extra memory for an operation, and for intermediate tensors coming from earlier layers and serving as arguments to later layers. The first kind of temporary memory can be easily retrieved from the profile. But the single-layer profile contains no model-level information. So we need to find this information ourselves.

For each layer invocation, we know the identity of each argument and the identity of each tensor output. For each tensor output, we can find the last layer using the tensor. Then for every layer between (and excluding) the layer generating the output tensor and the layer last using the output tensor, each layer in between needs to carry this intermediate tensor. 

That is to say, for each layer invocation
```python
  mem_intermediate[i] = sum(
    step_out_size[i]
    for j in range(n_layer)
    if step_out_firstalive[j]<=i
    and i<step_out_lastuse[j]
  )
```
Here we use a lazy update scheme. Instead of updating `mem_intermediate` for every tensor, we set a flag on two endpoints and do a final cumulative sum over the array.

Example for the lazy update scheme
```python
a = np.array([0,0,0,0,0,0])
a[0:2]+=2
a[1:3]+=4
```
is equivalent to
```python
a = np.array([0,0,0,0,0,0])
a[0]+=2
a[2]-=2
a[1]+=4
a[3]-=4
a = np.cumsum(a)
```

## Minimal example
Setup executable cache.
```bash
TF_POPLAR_FLAGS='--executable_cache_path=path_to_cache'
```
Run `AutoPipe` and compare the estimated memory and cycle with the actual memory and cycle.
```python
import tensorflow as tf
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.prediction import (
    partition, manual_test_utils)
from ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils import (
    types, types_utils)

def create_dataset_fake(n_batch,
                        batch_size,
                        x_shape,
                        y_max_range,
                        x_dtype=tf.float32,
                        y_dtype=tf.int32):
  dataset_size = n_batch * batch_size
  x_train = tf.random.uniform(
      shape=(dataset_size,) + x_shape, minval=0, maxval=1, dtype=x_dtype)
  y_train = tf.random.uniform(
      shape=(dataset_size,), minval=0, maxval=y_max_range, dtype=y_dtype)
  train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
      batch_size, drop_remainder=True)

  return train_ds

def create_model():
  return tf.keras.applications.ResNet50(
      include_top=True, weights="imagenet", pooling="avg")

def create_dataset(num_batch, batch_size):
  return create_dataset_fake(
      n_batch=num_batch,
      batch_size=batch_size,
      x_shape=(224, 224, 3),
      y_max_range=1000,
  )

# Describe the cluster
cluster_info = {
    "clock":
        1850000000,
    "memory":
        894,
    "connection": [{
        "recvGBps": 7,
        "recvIdleCycle": 50000,
        "sendGBps": 70,
        "sendIdleCycle": 3000
    }, {
        "recvGBps": 70,
        "recvIdleCycle": 3000,
        "sendGBps": 7,
        "sendIdleCycle": 50000
    }]
}

# Auto Pipeline configuration
config = {
    "memoryProportion": 0.85,
    "usePoplarEstimation": True
}

# Batch size of the model
batch_size = 16

# Profile the model
model_profile = partition.create_model_profile(create_model, batch_size, config)

# Get a partition
pipe_partition = partition.get_auto_pipeline_partition_by_cycle_and_transfer(
    model_profile, cluster_info, config)

# Print the partition
print("_".join([str(x) for x in pipe_partition]))

# Estimate the cost of the partition
est_range_info = partition.get_model_partition_cost(model_profile, cluster_info,
                                                    pipe_partition)

# Run and profile the model in pipeline
act_range_info = manual_test_utils.create_model_partitioned_profile(
    create_model, create_dataset, batch_size, pipe_partition)

# Print difference between estimation and actual
types_utils.compare_pipelined_model_profile(est_range_info, act_range_info)
```

# Auto Auto Pipeline Readme
While the `Auto Pipeline` tool is able to find the pipeline stage assignment, the algorithm still depends on some other hyper-parameters, such as batch size and other `IPUConfig`s. Can we automatically configure these hyperparameters to maximize the model inference performance?

The simplest way to find the optimal configuration is to perform a grid search. But grid search requires compiling the model many times.

We notice for some hyperparameters, such as batch size, a higher value means better performance, but also a higher risk of getting an OOM error. However, for some other hyper-parameters, such as the number of IPUs, a higher value does not necessarily mean better performance.

We can then binary search on one of the former hyper-parameter, after performing a grid search on all other hyperparameters.

## Example Code
In this example, we use binary search to find the maximum batch size, and use grid search to find an appropriate `n_ipu` and `availableMemoryProportion`.

```py
# Configuration for AutoPipe.
config = {"memoryProportion": 0.85, "usePoplarEstimation": False}

# Dict of configurations for binary search.
binary_search_options = {
  "batch_size": list(range(1,128))
}

# Dict of configurations for grid search.
grid_search_options = {
  "n_ipu": [1,2,4],
  "mem_portion": [0.3,0.6,1]
}

# A function to generate configurations for AutoPipe and Profiler from
# the two dictionaries of configurations.
def create_args(batch_size, n_ipu, mem_portion):
  # Set `availableMemoryProportion` in ipu_config.
  mem_str = str(mem_portion)
  ipu_config = ipu.config.IPUConfig()
  ipu_config.matmuls.poplar_options["availableMemoryProportion"] = mem_str

  return {
    # Use the default IPUConfig, with no extra configurations.
    "ipuConfig": ipu_config,

    # Batch size to use this time.
    "batchSize": batch_size,

    # AutoPipe configuration.
    "autoPipeConfig": config,

    # Description of the IPU system. Here `cluster_infos[n_ipu]` is the
    # `ipu_tensorflow_addons.keras.experimental.auto_pipeline.utils.types
    # .IPUClusterInfo` dictionary for a `n_ipu` system.
    "clusterInfo": cluster_infos[n_ipu]
  }

# Run the search.
max_throughput.search_for_max_throughput(binary_search_options,
                                   grid_search_options, create_args,
                                   create_model, create_dataset)
```