"""State Kernels: Using the technique of temporal differences to learn state representations for
reinforcement learning algorithms.

Usage:
    # Define your environment by implementing the StateKernelEnvironment interface.
    class MyEnvironment(StateKernelEnvironment):
        ...

    # Instantiate the environment.
    environment = MyEnvironment(...)

    # Instantiate the state kernel and add the appropriate kernel modules to optimize it for your
    # environment.
    kernel = StateKernel()
    kernel.add_module(...)
    ...

    # Create the configuration for the harness.
    config = StateKernelHarnessConfig(...)

    # Create the harness.
    harness = StateKernelHarness(config, kernel, environment)

    # Run the harness to make the state kernel and environment interact with each other.
    harness.run()

The classes provided by this library are thread-compatible but not thread safe. You can use the
same kernel in multiple concurrent harnesses on multiple environment instances, provided you
implement the appropriate locking.
TODO: Implement the model training locks in the library. It should be straight forward.
"""
