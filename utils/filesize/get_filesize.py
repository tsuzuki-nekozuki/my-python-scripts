import math

from pathlib import Path


def get_readable_size(
        size_in_bytes: int, unit: str | None = None) -> tuple[float, str]:
    """
    Return and with an optimized unit (B, KB, MB, GB, TB).

    Args:
        size_in_bytes (int): File size in bytes.
        unit (str | None, optional): Given unit. Defaults to None.

    Returns:
        tuple[float, str]: Tuple of (human-readable size, unit)
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_in_bytes)
    if (unit is None and unit not in units) or unit == '':
        i = 0
        while size >= 1024 and i < len(units) - 1:
            size /= 1024.0
            i += 1
        size = math.trunc(size * 10**2) / 10**2
    else:
        i = units.index(unit)
        size /= 1024.0 ** i
        if size >= 1.0:
            size = math.trunc(size * 10**2) / 10**2            
    return size, units[i]


def get_total_size_and_count(path: str | Path,
                             unit: str | None = None) -> tuple[str, int]:
    """
    Calculate the total size and file count for a file or directory.
    
    Args:
        path: File or directory path.
        unit (str | None, optional): Given unit. Defaults to None.
    
    Returns:
        Tuple of (human-readable size, file count).
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f'Path "{p}" does not exist.')

    if p.is_file():
        total_size = p.stat().st_size
        file_count = 1
    else:
        total_size = 0
        file_count = 0
        for f in p.rglob('*'):
            if f.is_file():
                total_size += f.stat().st_size
                file_count += 1

    return *get_readable_size(total_size, unit), file_count


if __name__ == '__main__':
    path_input = Path(input('Enter file or directory path: ').strip())
    unit_input = input('Enter file size unit: ').strip()
    try:
        size, unit, count = get_total_size_and_count(path_input, unit_input)
        print(f'Path: {path_input}')
        print(f'Total size: {size} {unit}')
        print(f'File count: {count:,}')
    except FileNotFoundError as e:
        print(e)
