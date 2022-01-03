"""State Kernels: Using the technique of temporal differences to learn state representations for
reinforcement learning algorithms.

Usage:
    # Define your sensors by implementing the InputProvider interface.
    class MySensor(InputProvider):
        ...

    # Define your task(s) by implementing the StateKernelModule interface.
    class MyTask(StateKernelModule):
        ...

    # Instantiate the sensor and task.
    sensor = MySensor(...)
    task = MyTask(...)

    # Instantiate the state kernel and add the appropriate kernel modules to optimize it for your
    # environment.
    kernel = StateKernel()
    kernel.add_module(sensor)
    kernel.add_module(task)
    kernel.add_module(...)
    ...

    # Configure the kernel
    config = StateKernelConfig(...)
    kernel.configure(config)

    # Create and run the state stream.
    stream = StateStream(kernel, environment)
    stream.run()

The classes provided by this library are thread-compatible but not thread safe. You can use the
same kernel in multiple concurrent harnesses on multiple environment instances, provided you
implement the appropriate locking.
TODO: Implement the model training locks in the library. It should be straight forward.
"""
