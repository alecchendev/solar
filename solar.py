import argparse
from enum import Enum

DATA_DIR = "data"


# Note: no Alaska or Hawaii
class State(Enum):
    ALABAMA = "al"
    ARKANSAS = "ar"
    ARIZONA = "az"
    CALIFORNIA = "ca"
    COLORADO = "co"
    CONNECTICUT = "ct"
    DELAWARE = "de"
    FLORIDA = "fl"
    GEORGIA = "ga"
    IDAHO = "id"
    ILLINOIS = "il"
    INDIANA = "in"
    IOWA = "ia"
    KANSAS = "ks"
    KENTUCKY = "ky"
    LOUISIANA = "la"
    MAINE = "me"
    MARYLAND = "md"
    MASSACHUSETTS = "ma"
    MICHIGAN = "mi"
    MINNESOTA = "mn"
    MISSISSIPPI = "ms"
    MISSOURI = "mo"
    MONTANA = "mt"
    NEBRASKA = "ne"
    NEVADA = "nv"
    NEW_HAMPSHIRE = "nh"
    NEW_JERSEY = "nj"
    NEW_MEXICO = "nm"
    NEW_MEXICO_EAST = "nm-east"
    NEW_YORK = "ny"
    NORTH_CAROLINA = "nc"
    OHIO = "oh"
    OKLAHOMA = "ok"
    OREGON = "or"
    PENNSYLVANIA = "pa"
    RHODE_ISLAND = "ri"
    SOUTH_CAROLINA = "sc"
    SOUTH_DAKOTA = "sd"
    TENNESSEE = "tn"
    TEXAS = "tx"
    TEXAS_EAST = "tx-east"
    UTAH = "ut"
    VERMONT = "vt"
    VIRGINIA = "va"
    WASHINGTON = "wa"
    WEST_VIRGINIA = "wv"
    WISCONSIN = "wi"
    WYOMING = "wy"


def add(a: int, b: int) -> int:
    return a + b


def main():
    parser = argparse.ArgumentParser(
        description="A tool to model cost and utilization of solar power systems."
    )
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # TODO: Download
    subparsers.add_parser("download", help="Download state solar datasets")

    # TODO: Produce optimal configurations
    subparsers.add_parser(
        "optimize",
        help="Produce optimal array and battery sizes for a range of load costs",
    )

    # TODO: Produce visuals
    subparsers.add_parser("plot", help="Visualize preset plots for datasets")

    # TODO: everything (optimize for a state's average plant, not literally every single one)
    subparsers.add_parser(
        "all", help="Download, optimize, and plot across all available datasets"
    )

    args = parser.parse_args()


if __name__ == "__main__":
    main()
