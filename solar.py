import argparse

DATA_DIR = "data"

# Note: no Alaska or Hawaii
EASTERN_STATES = [
    "al",
    "ar",
    "ct",
    "de",
    "fl",
    "ga",
    "il",
    "in",
    "ia",
    "ks",
    "ky",
    "la",
    "me",
    "md",
    "ma",
    "mi",
    "mn",
    "ms",
    "mo",
    "mt",
    "ne",
    "nh",
    "nj",
    "nm-east",
    "ny",
    "nc",
    "oh",
    "ok",
    "pa",
    "ri",
    "sc",
    "sd",
    "tn",
    "tx-east",
    "vt",
    "va",
    "wv",
    "wi",
]
WESTERN_STATES = [
    "az",
    "ca",
    "co",
    "id",
    "mt",
    "nv",
    "nm",
    "or",
    "sd",
    "tx",
    "ut",
    "wa",
    "wy",
]


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
