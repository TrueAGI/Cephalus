# Planned

* Create "task" and "sensor" module interfaces. Remove the environment
  interface. Tasks will subsume the role of current state gradient 
  providers (as opposed to future state gradient providers) by providing 
  loss signals, and sensors will take on the role of input providers 
  (possibly with trainable weights). Merge the harness into the kernel.
  The structure of the state model will need to be modified to allow for 
  an arbitrary number of sensors, using an attention mechanism.
* Make the modules list dynamic; users should be able to add or remove
  arbitrary modules on the fly, with the possible exception of the state
  and gradient prediction providers.
* Support multithreading.


# Completed
