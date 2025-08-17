import argparse
import io
import os
from typing import Any
import zipfile
from enum import Enum

import requests
import pandas as pd
import folium


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

# Plot state maps

US_STATE_CENTERS = {
    State.ALABAMA: (32.7794, -86.8287),
    State.ARKANSAS: (34.7990, -92.3747),
    State.ARIZONA: (34.2744, -111.2847),
    State.CALIFORNIA: (36.7783, -119.4179),
    State.COLORADO: (39.0646, -105.3272),
    State.CONNECTICUT: (41.5834, -72.7622),
    State.DELAWARE: (39.1612, -75.5264),
    State.FLORIDA: (27.7663, -81.6868),
    State.GEORGIA: (32.9866, -83.6487),
    State.IDAHO: (44.2394, -114.5103),
    State.ILLINOIS: (40.3363, -89.0022),
    State.INDIANA: (39.8647, -86.2604),
    State.IOWA: (42.0046, -93.2140),
    State.KANSAS: (38.5111, -96.8005),
    State.KENTUCKY: (37.6690, -84.6514),
    State.LOUISIANA: (31.1801, -91.8749),
    State.MAINE: (44.6074, -69.3977),
    State.MARYLAND: (39.0724, -76.7902),
    State.MASSACHUSETTS: (42.2373, -71.5314),
    State.MICHIGAN: (43.3504, -84.5603),
    State.MINNESOTA: (45.7326, -93.9196),
    State.MISSISSIPPI: (32.7673, -89.6812),
    State.MISSOURI: (38.4623, -92.302),
    State.MONTANA: (47.0527, -110.2148),
    State.NEBRASKA: (41.1289, -98.2883),
    State.NEVADA: (38.4199, -117.1219),
    State.NEW_HAMPSHIRE: (43.4108, -71.5653),
    State.NEW_JERSEY: (40.314, -74.5089),
    State.NEW_MEXICO: (34.8375, -106.2371),
    State.NEW_MEXICO_EAST: (34.8375, -103.0),
    State.NEW_YORK: (42.9538, -75.5268),
    State.NORTH_CAROLINA: (35.6411, -79.8431),
    State.OHIO: (40.3963, -82.7755),
    State.OKLAHOMA: (35.5376, -96.9247),
    State.OREGON: (44.5672, -122.1269),
    State.PENNSYLVANIA: (40.590752, -77.209755),
    State.RHODE_ISLAND: (41.6772, -71.5101),
    State.SOUTH_CAROLINA: (33.8191, -80.9066),
    State.SOUTH_DAKOTA: (44.2853, -99.4632),
    State.TENNESSEE: (35.7449, -86.7489),
    State.TEXAS: (31.106, -97.6475),
    State.TEXAS_EAST: (31.106, -94.0),
    State.UTAH: (40.1135, -111.8535),
    State.VERMONT: (44.0407, -72.7093),
    State.VIRGINIA: (37.768, -78.2057),
    State.WASHINGTON: (47.3917, -121.5708),
    State.WEST_VIRGINIA: (38.468, -80.9696),
    State.WISCONSIN: (44.2563, -89.6385),
    State.WYOMING: (42.7475, -107.2085),
}


def plot_state_map(files_df: pd.DataFrame, state: State) -> str:
    """Create a folium map showing solar plant locations for a state"""
    center = US_STATE_CENTERS[state]
    actual_state_df = files_df.loc[
        (files_df[DatasetColumn.DATA_TYPE] == DataType.ACTUAL)
        & (files_df[DatasetColumn.STATE] == state)
    ]

    m = folium.Map(location=center, zoom_start=7)
    coordinates = zip(
        actual_state_df[DatasetColumn.LATITUDE],
        actual_state_df[DatasetColumn.LONGITUDE],
    )
    for i, (lat, lon) in enumerate(coordinates):
        folium.Marker(
            [lat, lon],
            popup=f"Point {i + 1}: ({lat}, {lon})",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    filename = f"{state}_map.html"
    m.save(filename)
    return filename

# Main CLI

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
            elif not State.valid(args.state):
                print(
                    f"Error: {args.state} is not a valid state. Available states: {', '.join(State.all())}"
                )
            else:
                state = State.from_str(args.state)
                files_df = create_state_files_df(args.directory)
                filename = plot_state_map(files_df, state)
                print(
                    f"Map saved to file://{os.path.abspath(filename)} - open in browser to view (or run `open {filename}`)"
                )


if __name__ == "__main__":
    main()
