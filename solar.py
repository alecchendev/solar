import argparse
import io
import os
import zipfile
from enum import Enum

import requests

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

    def full_name(self) -> str:
        return self.name.replace("_", " ").title()

    @classmethod
    def from_str(cls, state_str: str):
        """Convert string (abbreviation or full name) to State enum"""
        state_str = state_str.lower()
        for state in cls:
            if state.value == state_str:
                return state

        # "new york" -> NEW_YORK
        enum_name = state_str.upper().replace(" ", "_")
        if hasattr(cls, enum_name):
            return cls[enum_name]
        raise ValueError(f"No state found for: {state_str}")

    @classmethod
    def valid(cls, state_str: str) -> bool:
        try:
            cls.from_str(state_str)
            return True
        except ValueError:
            return False

    @classmethod
    def all(cls) -> list[str]:
        return [state.value for state in cls]


def state_solar_zip_filename(state: str) -> str:
    return f"{state}-pv-2006.zip"


def state_download_url(state: str) -> str:
    return f"https://www.nrel.gov/docs/libraries/grid/{state_solar_zip_filename(state)}"


def state_data_dir(directory: str, state: str) -> str:
    return f"{directory}/{state}"


def download_state_solar_data(directory: str, state: State):
    print(f"Downloading solar data for {state.full_name()}...")
    os.makedirs(directory, exist_ok=True)
    url = state_download_url(state.value)
    extract_to = state_data_dir(directory, state.value)
    os.makedirs(extract_to, exist_ok=True)
    r = requests.get(url)
    print("Extracting files...")
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
    except Exception as e:
        print("Error when creating zip file from content:", e)
        return
    z.extractall(extract_to)
    print(f"Downloaded: {len(os.listdir(extract_to))} total")


def check_downloaded(directory: str, state: State) -> bool:
    state_dir = state_data_dir(directory, state.value)
    return os.path.isdir(state_dir) and len(os.listdir(state_dir)) > 0


def download_all_solar_data(directory: str, skip_existing: bool = True):
    for state in State:
        if check_downloaded(directory, state) and skip_existing:
            print(f"Already downloaded {state.full_name()}, skipping...")
            continue
        download_state_solar_data(directory, state)


def add(a: int, b: int) -> int:
    return a + b


def main():
    parser = argparse.ArgumentParser(
        description="A tool to model cost and utilization of solar power systems."
    )
    subparsers = parser.add_subparsers(title="commands", dest="command")

    download_parser = subparsers.add_parser(
        "download", help="Download state solar datasets"
    )
    download_parser.add_argument(
        "--state", default="", help="Download data for specific state (abbreviation)"
    )
    download_parser.add_argument(
        "--all", action="store_true", help="Download data for all states"
    )
    download_parser.add_argument(
        "--directory",
        default=DATA_DIR,
        help=f"Directory to download data to. Defaults to {DATA_DIR}",
    )

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

    if args.command == "download":
        if args.all == (args.state != ""):
            print("Error: Please specify either --state or --all")
            return
        if args.all:
            download_all_solar_data(args.directory)
            return
        if not State.valid(args.state):
            print(
                f"Error: {args.state} is not a valid state. Available states: {', '.join(State.all())}"
            )
        download_state_solar_data(args.directory, State.from_str(args.state))


if __name__ == "__main__":
    main()
