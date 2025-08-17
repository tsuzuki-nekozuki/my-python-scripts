import argparse
import random
import secrets
import string

def generate_random_string(length: int,
                           use_digits: bool,
                           use_upper: bool,
                           use_lower: bool,
                           use_symbols: bool,
                           use_others: bool,
                           secure: bool = False) -> str:
    """
    Generate a random string with specified character classes.
    
    Args:
        length (int): Length of the string.
        use_digits (bool): Include 0-9.
        use_upper (bool): Include A-Z.
        use_lower (bool): Include a-z.
        use_symbols (bool): Include punctuation symbols.
        use_others (bool): Include other half-width ASCII characters
                           (space, etc.).
        secure (bool): If True, use `secrets` for cryptographic randomness;
                       otherwise use `random`.
    
    Returns:
        str: Generated random string.
    """
    chars = ''
    if use_digits:
        chars += string.digits
    if use_upper:
        chars += string.ascii_uppercase
    if use_lower:
        chars += string.ascii_lowercase
    if use_symbols:
        chars += string.punctuation
    if use_others:
        chars += ''  # You can add anything you want
    if not chars:
        raise ValueError("At least one character set must be selected.")

    result = []
    for _ in range(length):
        if secure:
            result.append(secrets.choice(chars))
        else:
            result.append(random.choice(chars))
    return ''.join(result)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate random string with customizable character sets.")
    parser.add_argument('-l', '--length', type=int, default=12, help='Length of the generated string.')
    parser.add_argument('-d', '--digits', action='store_true', help='Include digits 0-9.')
    parser.add_argument('-u', '--upper', action='store_true', help='Include uppercase letters A-Z.')
    parser.add_argument('-L', '--lower', action='store_true', help='Include lowercase letters a-z.')
    parser.add_argument('-s', '--symbols', action='store_true', help='Include punctuation symbols.')
    parser.add_argument('-o', '--others', action='store_true', help='Include other half-width ASCII characters (e.g., space).')
    parser.add_argument('--secure', action='store_true', help='Use cryptographically secure randomness (secrets).')
    return parser.parse_args()


def main():
    args = parse_args()
    rand_str = generate_random_string(
        length=args.length,
        use_digits=args.digits,
        use_upper=args.upper,
        use_lower=args.lower,
        use_symbols=args.symbols,
        use_others=args.others,
        secure=args.secure
    )
    print(rand_str)


if __name__ == "__main__":
    main()
