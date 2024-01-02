def location_to_index(loc: str) -> tuple[int, int]:
    """Convert a location string, e.g. "A1" or "P12", to a zero-indexed tuple, e.g. (0, 0)
    """
    assert 2 <= len(loc) <= 3
    letter = loc[0]
    number = int(loc[1:])

    assert letter in "ABCDEFGHIJKLMNOP"

    i = number - 1
    j = ord(letter) - ord("A")

    return i, j


def index_to_location(i: int, j: int) -> str:
    """Convert a zero-indexed tuple, e.g. (0, 0), to a location string, e.g. "A1"
    """
    assert 0 <= i <= 16
    assert 0 <= j <= 24

    letter = chr(ord("A") + j)
    number = i + 1

    return f"{letter}{number}"


def spreadsheet_plate_to_numeric(plate: str) -> int:
    """Convert a plate string, e.g. "Plate 01", to a numeric value, e.g. 1
    """
    assert plate.startswith("Plate ")
    return int(plate[6:])


def parse_name(f):
    f = str(f)
    parts = f.split(" ")
    parts = parts[1].split("-")

    plate_num = int(parts[0])
    measurement_num = parts[1]

    return plate_num, measurement_num
