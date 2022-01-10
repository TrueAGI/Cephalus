# Design Goals

* The "keras" of reinforcement learning:
  * Fully modular design.
  * Plug 'n' play: Works out of the box with predefined components and minimal configuration.
  * Fully customizable.
* Focus on on-line learning, but supports off-line learning.
* Supports distributed learning: Multiple instances that share the same models but have their
  own state and operate in distinct environments.
* Supports multitask learning: Agents that operate within the same environment and share 
  information.
* Supports dynamic reconfiguration of modules, including observation spaces, action spaces, and
  agents, midstream during execution.
* Full persistence, including not only models but state information of running kernels, to enable
  stopping and resuming processes mid-stream.
