# Planned

* Get rid of `GradientProvider`. Instead, measure the state gradient
  directly via the gathered losses at some point after the previous
  state is updated. All gradients should be provided via the tape.
* Support multithreading.


# Completed

* Make the modules list dynamic; users should be able to add or remove
  arbitrary modules on the fly, with the possible exception of the state
  and gradient prediction providers.
* Create "task" and "sensor" module interfaces. Remove the environment
  interface. Tasks will subsume the role of current state gradient 
  providers (as opposed to future state gradient providers) by providing 
  loss signals, and sensors will take on the role of input providers 
  (possibly with trainable weights). Merge the harness into the kernel.
  The structure of the state model will need to be modified to allow for 
  an arbitrary number of sensors, using an attention mechanism. **What
  I actually did:** Created `GradientProvider` and `InputProvider` for
  task and sensor modules, respectively. Renamed `StateKernelHarness`
  to `StateStream` and moved most of its functionality to `StateKernel`,
  leaving it with only the task of keeping track of state frames so
  the state kernel remains stateless w.r.t. a given state stream. Added
  the attention provider module type to combine an arbitrary number of
  inputs received from the input providers into a single signal for the
  purpose of updating the state.
