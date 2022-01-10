# Cephalus: TODO

---

* [Source]
* [Readme]
* [License]
* [Design Goals]
* [TODO] (this document)

[Source]: https://github.com/TrueAGI/Cephalus
[Readme]: README.md
[License]: LICENSE.md
[Design Goals]: Design%20Goals.md
[TODO]: TODO.md

---

## Planned Changes

* Unit tests!
* Performance testing: Processing speed, and speed, consistency, and stability 
  of learning. (Others?)
* Model checkpoints and persistence.
* Better/more documentation, including:
  * Move the 'A Note on Adaptation of Policy Gradients for State 
    Representation Induction' section in [README.md] to a Jupyter notebook
    and actually demonstrates (1) how it works and (2) the advantages of
    the particular design decisions made.
  * Document each component thoroughly: What it is and when/where/why/how to 
    use it.
* History-augmented state. Options must include both simple concatenation and
  attention mechanisms.
* Training sample collection, caching and reuse. This will require not only
  collecting the training targets from all the modules, but passing them back
  again when loss is computed during training.
* Support multithreading explicitly. Thread "compatibility" is not enough.
* Normalization of sensor inputs, either automatically with on-line statistics
  or using user-provided ranges or mappings.
* Do some experiments to see if it's a good idea to set the discount rate
  dynamically based on the prediction error of the q model. (And likewise,
  the loss scale for the TD state loss based on its prediction error.)
  The idea is that the lower the prediction error of the q model, the 
  longer range we can look forward without noise swamping the signal. This
  horizon should gradually expand with time, in lockstep with the model's 
  ability to support it.
* Research prior work on the various algorithms I have implemented or plan 
  to implement in cephalus.modules.retroactive_loss to determine which I
  can actually claim as my own inventions, which I can claim as novel
  applications to a new domain (state representation induction, as opposed
  to action selection), and which I can cite existing research for because 
  they aren't novel after all.

## Completed Changes

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
* Get rid of `GradientProvider`. Instead, measure the state gradient
  directly via the gathered losses at some point after the previous
  state is updated. All gradients should be provided via the tape.
  **What I actually did**: Renamed `RetroactiveGradientProvider` to 
  `RetroactiveLossProvider` and refactored the kernel to use only
  gradients derived from module losses.
* Add a license. **What I did:** Added the Apache 2.0 license. 
* The @sensor decorator doesn't allow for the same sensor mapping to be reused 
  for multiple environments. If we want to reuse the same kernel with multiple 
  environments of the same type, it will have to relearn the sensor mappings 
  for each of them, even though they are actually the same. So how do we modify 
  the @sensor decorator to remember the mapping and share it across sensors of 
  the same type? We can map them in a global dictionary, keyed by the functions 
  they wrap. There will need to be methods for marking a sensor as single-use 
  and for retiring a sensor that will no longer be reused.




[README.md]: README.md
