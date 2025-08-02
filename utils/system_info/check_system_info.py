import platform
import socket
import shutil
import subprocess

import cpuinfo
import psutil


def get_os_info() -> dict:
    """
    Get operating system and kernel information.

    Returns:
        dict: Contains keys 'OS', 'OS Version', 'Kernel', 'Architecture'.
    """
    return {
        'OS': platform.system(),
        'OS Version': platform.version(),
        'Kernel': platform.release(),
        'Architecture': platform.machine(),
    }


def get_cpu_info() -> dict:
    """
    Get CPU and core count information.

    Returns:
        dict: Contains keys 'CPU', 'Cores (Physical)', 'Cores (Logical)'.
    """
    return {
        'CPU': cpuinfo.get_cpu_info().get('brand_raw'),
        'Cores (Physical)': psutil.cpu_count(logical=False),
        'Cores (Logical)': psutil.cpu_count(logical=True),
    }


def get_memory_info() -> dict:
    """
    Get system memory information.

    Returns:
        dict: Contains key 'Memory Total (GB)' with float value.
    """
    mem = psutil.virtual_memory()
    return {
        'Memory Total (GB)': round(mem.total / (1024 ** 3), 2)
    }


def get_nvidia_gpu_info() -> dict:
    """
    Retrieve information about NVIDIA GPUs using the `nvidia-smi` command.

    This function queries the system's NVIDIA GPUs (if available) and collects:
      - The number of detected GPUs
      - The name of each GPU
      - The total memory of each GPU (converted to GB if larger than 1 MiB)

    The information is returned in a dictionary with the following keys:
        - 'Number of Nvidia GPU' (int)
        - 'GPU0 name', 'GPU1 name', ... (str)
        - 'GPU0 memory_total', 'GPU1 memory_total', ... (str, e.g. '8.0 GB')

    Returns:
        dict: A dictionary containing GPU information. 
              Returns an empty dictionary if `nvidia-smi` is not available
              or fails to execute.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=name,memory.total',
             '--format=csv,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        gpus = {}
        gpus['Number of Nvidia GPU'] = len(lines)
        for i, iline in enumerate(lines):
            name, mem = iline.split(',')
            mem_display = mem.strip()
            size_str, unit = mem_display.split(' ')
            size = round(int(size_str) / 1024, 2)
            if size > 1 and unit == 'MiB':
                mem_display = '{} GB'.format(size)
            gpus['GPU{} name'.format(i)] = name.strip()
            gpus['GPU{} memory_total'.format(i)] = mem_display
        return gpus
    except subprocess.CalledProcessError:
        return {}


def get_disk_info() -> dict:
    """
    Get total and used disk space for the root filesystem.

    Returns:
        dict: Contains keys 'Disk Total (GB)', 'Disk Used (GB)'
              and 'Disk Free (GB)'.
    """
    total, used, free = shutil.disk_usage('/')
    return {
        'Disk Total (GB)': round(total / (1024 ** 3), 2),
        'Disk Used (GB)': round(used / (1024 ** 3), 2),
        'Disk Free (GB)': round(free / (1024 ** 3), 2)       
    }


def get_network_info() -> dict:
    """
    Get network hostname and IP addresses for all interfaces.

    Returns:
        dict: Keys include 'Hostname', '<interface> IPv4',
              and '<interface> IPv6'.
    """
    info = {'Hostname': socket.gethostname()}
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                info[f'{interface} IPv4'] = addr.address
            elif addr.family == socket.AF_INET6:
                info[f'{interface} IPv6'] = addr.address
    return info


def get_python_info() -> dict:
    """
    Get Python interpreter information.

    Returns:
        dict: Contains keys 'Python Version' and 'Python Executable'.
    """
    return {
        'Python Version': platform.python_version(),
        'Python Executable': shutil.which('python'),
    }


def get_system_info() -> dict:
    """
    Aggregate system information including OS, CPU, memory, disk, GPU, network,
    and Python.

    Returns:
        dict: Consolidated system information with human-readable values.
    """
    info = {}
    info.update(get_os_info())
    info.update(get_cpu_info())
    info.update(get_memory_info())
    info.update(get_nvidia_gpu_info())
    info.update(get_disk_info())
    info.update(get_network_info())
    info.update(get_python_info())
    return info


if __name__ == '__main__':
    system_info = get_system_info()
    for k, v in system_info.items():
        print(f'{k}: {v}')
