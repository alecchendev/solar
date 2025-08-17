import argparse
import io
import os
from typing import Any
import zipfile
from enum import Enum

import requests
import pandas as pd


class StringEnum(str, Enum):
    """Enum for defining a group of string constants."""

    def __str__(self) -> str:
        return self.value


# Note: no Alaska or Hawaii
class State(StringEnum):
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


# Download


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


# File metadata


class DataType(StringEnum):
    ACTUAL = "Actual"  # Real power output
    DA = "DA"  # Day ahead forecast
    HA4 = "HA4"  # 4 hour ahead forecast


class PvType(StringEnum):
    UPV = "UPV"  # Utility scale PV
    DPV = "DPV"  # Distributed PV


class DatasetColumn(StringEnum):
    STATE = "state"
    DATA_TYPE = "data_type"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    WEATHER_YEAR = "weather_year"
    PV_TYPE = "pv_type"
    CAPACITY_MW = "capacity_mw"
    TIME_INTERVAL_MIN = "time_interval_min"


def solar_filename(
    data_type: DataType,
    latitude: float,
    longitude: float,
    weather_year: int,
    pv_type: PvType,
    capacity_mw: float,
    time_interval_min: int,
) -> str:
    return f"{data_type}_{round(latitude, 2)}_{round(longitude, 2)}_{weather_year}_{pv_type}_{int(capacity_mw) if capacity_mw.is_integer() else capacity_mw}MW_{time_interval_min}_Min.csv"


def filename_from_dict(d: dict[str, Any]) -> str:
    return solar_filename(
        **{str(col): d[col] for col in DatasetColumn if col != DatasetColumn.STATE}
    )


def metadata_from_filename(state: State, filename: str) -> dict[str, Any]:
    name = filename.removesuffix(".csv")
    parts = name.split("_")
    assert len(parts) == 8  # 8 because of "5_Min" being two parts
    (
        data_type_str,
        lat_str,
        lon_str,
        year_str,
        pv_type_str,
        capacity_str,
        interval_val,
        interval_label,
    ) = parts
    assert interval_label == "Min"
    return {
        DatasetColumn.STATE: state,
        DatasetColumn.DATA_TYPE: DataType(data_type_str),
        DatasetColumn.LATITUDE: float(lat_str),
        DatasetColumn.LONGITUDE: float(lon_str),
        DatasetColumn.WEATHER_YEAR: int(year_str),
        DatasetColumn.PV_TYPE: PvType(pv_type_str),
        DatasetColumn.CAPACITY_MW: float(capacity_str[:-2]),  # chop the MW off "5MW"
        DatasetColumn.TIME_INTERVAL_MIN: int(interval_val),
    }


def create_state_files_df(directory: str) -> pd.DataFrame:
    rows = []
    for state in State:
        assert check_downloaded(directory, state)
        files = os.listdir(state_data_dir(directory, state.value))
        state_rows = [metadata_from_filename(state, file) for file in files]
        rows += state_rows
    return pd.DataFrame(rows)


class Command(StringEnum):
    DOWNLOAD = "download"
    OPTIMIZE = "optimize"
    PLOT = "plot"


DEFAULT_DATA_DIRECTORY = "data"


def main():
    parser = argparse.ArgumentParser(
        description="A tool to model cost and utilization of solar power systems."
    )
    subparsers = parser.add_subparsers(title="commands", dest="command")

    download_parser = subparsers.add_parser(
        Command.DOWNLOAD, help="Download state solar datasets"
    )
    download_parser.add_argument(
        "--state", default="", help="Download data for specific state (abbreviation)"
    )
    download_parser.add_argument(
        "--all", action="store_true", help="Download data for all states"
    )
    download_parser.add_argument(
        "--directory",
        default=DEFAULT_DATA_DIRECTORY,
        help=f"Directory to download data to. Defaults to `{DEFAULT_DATA_DIRECTORY}`",
    )

    # TODO: Produce optimal configurations
    subparsers.add_parser(
        Command.OPTIMIZE,
        help="Produce optimal array and battery sizes for a range of load costs",
    )

    # TODO: Produce visuals
    plot_parser = subparsers.add_parser(
        Command.PLOT, help="Visualize preset plots for datasets"
    )
    plot_parser.add_argument(
        "--kind",
        required=True,
        help=f"",
    )
    plot_parser.add_argument(
        "--state", default="", help="Which state to plot a map of solar plants."
    )
    plot_parser.add_argument(
        "--directory",
        default=DEFAULT_DATA_DIRECTORY,
        help=f"Directory to download data to. Defaults to `{DEFAULT_DATA_DIRECTORY}`",
    )

    # TODO: everything (optimize for a state's average plant, not literally every single one)
    subparsers.add_parser(
        "all", help="Download, optimize, and plot across all available datasets"
    )

    args = parser.parse_args()

    if args.command == Command.DOWNLOAD:
        if args.all == (args.state != ""):
            print("Error: Please specify either --state or --all")
        elif args.all:
            download_all_solar_data(args.directory)
        elif not State.valid(args.state):
            print(
                f"Error: {args.state} is not a valid state. Available states: {', '.join(State.all())}"
            )
        else:
            download_state_solar_data(args.directory, State.from_str(args.state))
    elif args.command == Command.OPTIMIZE:
        pass
    elif args.command == Command.PLOT:
        if args.kind == "map":
            if args.state == "":
                print("Error: Please specify --state to plot")
            else:
                pass


if __name__ == "__main__":
    main()
