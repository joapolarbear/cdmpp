import numpy as np
import subprocess as sp

from .device_info import short_device_name

def get_gpu_info(query_option):
    '''
    Some help info with the command `nvidia-smi --help-query-gpu`:
        
        ...
        
        Section about memory properties
        On-board memory information. Reported total memory is affected by ECC state. If ECC is enabled the total available memory is decreased by several percent, due to the requisite parity bits. The driver may also reserve a small amount of memory for internal use, even without active work on the GPU.

        "memory.total"
        Total installed GPU memory.

        "memory.used"
        Total memory allocated by active contexts.

        "memory.free"
        Total free memory.

        ...

        Section about utilization properties
        Utilization rates report how busy each GPU is over time, and can be used to determine how much an application is using the GPUs in the system.

        "utilization.gpu"
        Percent of time over the past sample period during which one or more kernels was executing on the GPU.
        The sample period may be between 1 second and 1/6 second depending on the product.

        "utilization.memory"
        Percent of time over the past sample period during which global (device) memory was being read or written.
        The sample period may be between 1 second and 1/6 second depending on the product.
        
        ...

        Section about clocks.max properties
        Maximum frequency at which parts of the GPU are design to run.

        "clocks.max.graphics" or "clocks.max.gr"
        Maximum frequency of graphics (shader) clock.

        "clocks.max.sm" or "clocks.max.sm"
        Maximum frequency of SM (Streaming Multiprocessor) clock.

        "clocks.max.memory" or "clocks.max.mem"
        Maximum frequency of memory clock.

        ...

    '''
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = f"nvidia-smi --query-gpu={query_option} --format=csv"
    try:
        help_info = output_to_list(sp.check_output(COMMAND.split(), stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    return help_info

def get_gpu_memory_in_use():
    help_info = get_gpu_info("memory.used")
    return [int(x.split()[0]) for i, x in enumerate(help_info)]

def get_gpu_util():
    help_info = get_gpu_info("utilization.gpu")
    return [int(x.split()[0]) for i, x in enumerate(help_info)]

def get_gpu_name():
    ''' Get GPU names
    Possible returned name: 
    Tesla V100-PCIE-32GB, 
    '''
    help_info = get_gpu_info("name")
    return [short_device_name(x.replace(" ", "_")) for i, x in enumerate(help_info)]

def check_gpu_util():
    gpu_util = get_gpu_util()
    if np.average(gpu_util) > 20:
        raise ValueError("The GPU is busy with the utilization = {} %".format(gpu_util))
