def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert a hex color string (e.g., 'FFEC09' or '#FFEC09') into an RGB tuple.

    Args:
        hex_color (str): Hex color string.

    Raises:
        ValueError: If the Hex color format is wrong.
        ValueError: If the Hex color format contains wrong characters.

    Returns:
        tuple[int, int, int]: RGB color tuple.
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError('Hex color must be 6 characters long')

    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except ValueError as exc:
        raise ValueError(
            f'"Hex color {hex_color}" contains invalid characters.') from exc
    return (r, g, b)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """
    Convert an RGB tuple into a hex color string.

    Args:
        rgb (tuple[int, int, int]): RGB color scale.

    Raises:
        ValueError: If RGB values are not in the range 0-255.

    Returns:
        str: Hex color string.
    """
    if any(not (0 <= c <= 255) for c in rgb):
        raise ValueError(f"RGB values {rgb} must be in the range 0-255")
    return "#{:02X}{:02X}{:02X}".format(*rgb)


if __name__ == '__main__':
    print('1. Hex -> RGB')
    print('2. RGB -> Hex')
    mode = input('Choose mode (1 or 2): ')
    if mode == '1':
        color_hex = input('Hex Color: ')
        rgb = hex_to_rgb(color_hex)
        print(rgb)
    elif mode == '2':
        rgb_values = input('RGB values (R,G,B): ')
        try:
            rgb = [int(i) for i in rgb_values.split(',')]
        except ValueError:
            print('Wrong RGB values.')
            exit()
        if len(rgb) == 3:
            rgb = tuple(rgb)
        else:
            print('RGB values must have 3 integers.')
            exit()            
        hex_color = rgb_to_hex(rgb)
        print(hex_color)
    else:
        print('Wrong mode')

