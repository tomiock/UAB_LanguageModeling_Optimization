# Memory Profiling and torch.compile — Usage Guide

## Memory profiling snapshots

### What gets captured

Three training scripts record a full GPU memory timeline and save it to a `.pickle` file:

| Script | Snapshot file |
|---|---|
| `finetune_summarization.py` | `memory_snapshot.pickle` |
| `finetune_full_epoch.py` | `memory_snapshot_epoch.pickle` |
| `finetune_half_two_epochs.py` | `memory_snapshot_half2e.pickle` |

Recording starts at optimizer step `PROFILE_AT_STEP` (default: 3) and runs for `PROFILE_NUM_STEPS` (default: 3) optimizer steps. With `gradient_accumulation_steps=4`, each optimizer step contains 4 micro-steps, so the snapshot covers 12 micro-steps total. Step 3 is chosen as the start because the first two steps include dataloader spin-up and (if `torch.compile` is on) the compilation warmup, which would distort the memory picture.

The recording API used is:

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)  # start
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")      # save
torch.cuda.memory._record_memory_history(enabled=None)          # stop
```

### Console memory table

When the snapshot is complete, the trainer prints a table to stdout:

```
step     μ   bef-fwd   aft-fwd    pk-fwd   aft-bwd    pk-bwd  (GB)
--------------------------------------------------------------------
   3     1     4.821     9.134    11.203     7.652    13.018
   3     2     7.652     9.134    11.203     7.652    13.018
   ...
```

Column meanings:

| Column | What it measures |
|---|---|
| `bef-fwd` | Memory before the forward pass: model weights + optimizer states |
| `aft-fwd` | After forward: adds activation tensors cached for the backward pass |
| `pk-fwd` | Peak during forward: adds intermediate tensors that are freed before backward |
| `aft-bwd` | After backward: adds gradient tensors, releases cached activations |
| `pk-bwd` | Peak during backward: worst-case allocation during the backward pass |

The gap `pk-fwd − aft-fwd` is the cost of intermediate tensors. The gap `aft-bwd − bef-fwd` is the gradient memory overhead. These numbers are useful for understanding where VRAM pressure comes from.

### Viewing the snapshot in the browser

Go to **https://pytorch.org/memory_viz** and drag the `.pickle` file onto the page. The visualizer shows:

- A timeline of every `malloc` and `free` event on the GPU.
- A flame-graph style breakdown of which Python call stacks allocated the largest tensors.
- Filtering by time range, allocation size, and call site.

The timeline makes it easy to see the activation memory spike during the forward pass and how it collapses after the backward pass completes.

### Tuning the profiling window

The three constants at the top of each script control the profiling:

```python
PROFILE_AT_STEP   = 3   # first optimizer step to record (0-indexed from 1)
PROFILE_NUM_STEPS = 3   # how many optimizer steps to capture
```

Set `PROFILE_AT_STEP` higher if you want to capture steady-state memory after the LR warmup has finished. Reduce `PROFILE_NUM_STEPS` to 1 if you only need a single representative step. Set `PROFILE_MEMORY = False` to skip the snapshot entirely and remove its overhead.