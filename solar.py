import argparse
import io
import os
from typing import Any
import zipfile
from enum import Enum

import requests
import pandas as pd
import folium
import numpy as np
import matplotlib.pyplot as plt


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


def check_downloaded(directory: str, state: State) -> bool:
    state_dir = state_data_dir(directory, state.value)
    return os.path.isdir(state_dir) and len(os.listdir(state_dir)) > 0


def download_state_solar_data(directory: str, state: State, skip_existing: bool = True):
    if check_downloaded(directory, state) and skip_existing:
        print(f"Already downloaded solar data for {state.full_name()}, skipping...")
        return
    print(f"Downloading solar data for {state.full_name()}...", end="\r")
    os.makedirs(directory, exist_ok=True)
    url = state_download_url(state.value)
    extract_to = state_data_dir(directory, state.value)
    os.makedirs(extract_to, exist_ok=True)
    r = requests.get(url)
    print("Extracting files...", end="\r")
    try:
        z = zipfile.ZipFile(io.BytesIO(r.content))
    except Exception as e:
        print("Error when creating zip file from content:", e)
        return
    z.extractall(extract_to)
    print(
        f"Downloaded solar data for {state.full_name()}: {len(os.listdir(extract_to))} total"
    )


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


def create_state_files_df(directory: str, states: list[State]) -> pd.DataFrame:
    rows = []
    for state in states:
        assert check_downloaded(directory, state)
        files = os.listdir(state_data_dir(directory, state.value))
        state_rows = [metadata_from_filename(state, file) for file in files]
        rows += state_rows
    return pd.DataFrame(rows)


# Optimization


class PowerColumn(StringEnum):
    LOCAL_TIME = "local_time"
    POWER_MW = "power_mw"


def read_plant_csv(directory: str, state: State, filename: str) -> pd.DataFrame:
    return pd.read_csv(f"{directory}/{state}/{filename}").rename(
        columns={"LocalTime": PowerColumn.LOCAL_TIME, "Power(MW)": PowerColumn.POWER_MW}
    )


def uptime_with_battery_with_inputs(
    load: float, battery_sizes: np.ndarray, array_sizes: np.ndarray, sol: np.ndarray
) -> pd.DataFrame:
    """
    Simulates battery storage system to compute uptime and utilization.

    Here load is a normalization parameter, i.e. `sol` is assumed to be normalized
    to `load` MW, and we can compute uptime and utilization of many battery +
    array sizes, without having to simulate many scaled versions of `sol`.
    I think there is a more intuitive way to write this (especially given
    we're in python land and not mathematica, but I haven't gotten around
    to it).

    An assumption here is that load is constant throughout a sim, when in
    reality it fluctuates. It would be really cool to get a load data set to fill
    in the variability across each part of the day/year.

    Can also simulate as no battery system if battery_sizes are zero.

    Parameters:
    battery_sizes: List of battery capacities (xWh).
    array_sizes: List of solar array sizes (xW).
    sol: Solar generation (xW) over time steps.

    Important thing here is that all parameters are in the same kW vs mW vs etc. unit.

    Returns:
    pd.DataFrame with the following columns:
        - Battery capacities
        - Load demands
        - Uptime (fraction of time battery is non-empty)
        - Utilization (fraction of load met)
    """
    # Initialize arrays
    n_steps = len(sol)
    n_battery_sizes = len(battery_sizes)
    n_array_sizes = len(array_sizes)
    time_step = 8760 / n_steps  # Time step in hours since capacity is in xWh

    # 2D matrices for loads and capacities
    # Normalize by array_sizes to avoid scaling many sols for each array_size
    loadsmat = np.tile(load / array_sizes, (n_battery_sizes, 1))
    capsmat = np.tile(battery_sizes[:, np.newaxis], (1, n_array_sizes)) / loadsmat

    array_size_mat = np.tile(array_sizes, (n_battery_sizes, 1))
    battery_size_mat = np.tile(battery_sizes[:, np.newaxis], (1, n_array_sizes))

    # 3D array for battery state: [time steps, capacities, loads]
    batt = np.zeros((n_steps, n_battery_sizes, n_array_sizes))
    batt[0] = capsmat  # Batteries start full

    # 3D array for utilization, initialized to zeros
    util = np.zeros_like(batt)

    # Main loop over time steps
    for i in range(n_steps - 1):
        # Solar discretization: 1 if solar >= load, 0 otherwise
        # sundisc = 0.5 + 0.5 * np.sign(sol[i] - loadsmat)
        sundisc = (sol[i] >= loadsmat).astype(int)

        # Battery discharge discretization: 1 if battery can cover shortfall
        # Make sure that we don't accidentally add to battery if solar exceeds load
        # battdisc = 0.5 + 0.5 * np.sign(batt[i] - time_step * (loadsmat - sol[i]))
        battdisc = (batt[i] >= time_step * np.maximum(loadsmat - sol[i], 0.0)).astype(
            int
        )

        # Utilization update
        util[i] = (
            sundisc
            + (1 - sundisc) * battdisc
            + (1 - sundisc)
            * (1 - battdisc)
            * (sol[i] / loadsmat + batt[i] / (time_step * loadsmat))
        )

        # Battery state update
        batt[i + 1] = (
            sundisc * (batt[i] + time_step * (sol[i] - loadsmat))
            + (1 - sundisc) * battdisc * (batt[i] - time_step * (loadsmat - sol[i]))
            + (1 - sundisc) * (1 - battdisc) * 0.0
        )  # 0 case here just to explicitly show else clause

        # Apply capacity constraint
        # NOTE: important difference between this and reference,
        # we need to compute capdisc on the next battery state not the current.
        # This fixes a bug where we get "uptime" even when we pass in
        # a battery with capacity 0.
        batt[i + 1] = np.minimum(batt[i + 1], capsmat)

    # Compute uptime (fraction of time battery is non-empty)
    uptime = np.mean(np.sign(batt), axis=0)

    # Compute utilization (fraction of load met)
    utilization = np.mean(util, axis=0)

    # Create DataFrame
    # Flatten capsmat, loadsmat, uptime, and utilization for DataFrame
    battery_size_flat = battery_size_mat.flatten()
    array_size_flat = array_size_mat.flatten()
    caps_flat = capsmat.flatten()
    loads_flat = loadsmat.flatten()
    uptime_flat = uptime.flatten()
    utilization_flat = utilization.flatten()

    # Construct DataFrame
    df = pd.DataFrame(
        {
            "battery_size": battery_size_flat,
            "array_size": array_size_flat,
            "capacity": caps_flat,
            "load": loads_flat,
            "uptime": uptime_flat,
            "utilization": utilization_flat,
        }
    )

    return df


def all_in_system_cost(
    solar_cost: float,
    battery_cost: float,
    load_cost: float,
    capacity: np.ndarray,
    load: np.ndarray,
    utilization: np.ndarray,
) -> np.ndarray:
    return (capacity * battery_cost + solar_cost + load_cost * load) / (
        load * utilization
    )


def all_in_system_cost_parallel(
    solarcost: float,
    batterycost: float,
    loadcost: float,
    load: float,
    batterysize: np.ndarray,
    arraysize: np.ndarray,
    sol: np.ndarray,
) -> pd.DataFrame:
    result = uptime_with_battery_with_inputs(load, batterysize, arraysize, sol)
    result["cost"] = all_in_system_cost(
        solarcost,
        batterycost,
        loadcost,
        result["capacity"].to_numpy(),
        result["load"].to_numpy(),
        result["utilization"].to_numpy(),
    )
    return result


class OptimizeColumn(StringEnum):
    SOLAR_COST_MW = "solar cost $/MW"
    BATTERY_COST_MWH = "battery cost $/MWh"
    LOAD_COST_MW = "load cost $/MW"
    ARRAYSIZE_MW = "arraysize (MW)"
    BATTERY_SIZE_MWH = "battery size (MWh)"
    LOAD_SIZE_MW = "load size (1 MW by definition)"
    ARRAY_COST = "array cost $"
    BATTERY_COST = "battery cost $"
    LOAD_COST_NORMALIZED = "load cost $ (all normalized to 1 MW)"
    TOTAL_POWER_SYSTEM_COST = "total power system cost $"
    TOTAL_SYSTEM_COST = "total system cost $"
    TOTAL_SYSTEM_COST_PER_UTILIZATION = "total system cost per utilization"
    BATTERY_SIZE_RELATIVE = "battery size relative to 1 MW array"
    LOAD_SIZE_RELATIVE = "load size relative to 1 MW array"
    ANNUAL_BATTERY_UTILIZATION = "annual battery utilization"
    ANNUAL_LOAD_UTILIZATION = "annual load utilization"


START_ARRAY_DIM = 30.0
START_BATTERY_DIM = 30.0


def find_minimum_system_cost_parallel(
    solar_cost: float,
    battery_cost: float,
    load_cost: float,
    load: float,
    sol: np.ndarray,
) -> dict[str, Any]:
    # Tweak these parameters to trade speed vs. optimality
    array_dim = START_ARRAY_DIM
    battery_dim = START_BATTERY_DIM
    array_size = START_ARRAY_DIM / 2  # start search from 0 to array_dim
    battery_size = START_BATTERY_DIM / 2  # start search from 0 to battery_dim
    array_density = 1.0
    battery_density = 1.0
    zoom_factor = 10  # how much to shrink search dim + density each step
    n_steps = 4

    min_cost = None
    best_row = None

    for _ in range(n_steps):
        start_array = array_size - array_dim / 2
        end_array = array_size + array_dim / 2
        array_sizes = np.arange(
            max(start_array, array_density), end_array, array_density
        )

        start_battery = battery_size - battery_dim / 2
        end_battery = battery_size + battery_dim / 2
        battery_sizes = np.arange(max(start_battery, 0), end_battery, battery_density)

        costs = uptime_with_battery_with_inputs(load, battery_sizes, array_sizes, sol)
        costs["cost"] = all_in_system_cost(
            solar_cost,
            battery_cost,
            load_cost,
            costs["capacity"].to_numpy(),
            costs["load"].to_numpy(),
            costs["utilization"].to_numpy(),
        )

        best_rows = costs.loc[costs["cost"] == costs["cost"].min()]
        best_row = best_rows.iloc[0]
        # Add a small amount to accept a tolerance for floating point shenanigans
        assert min_cost is None or best_row["cost"] <= min_cost + 0.0000001, (
            f"{min_cost} {best_row['cost']} > {min_cost}, {battery_size} != {best_row['battery_size']}, {array_size} != {best_row['array_size']}"
        )
        min_cost = best_row["cost"]
        battery_size = best_row["battery_size"]
        array_size = best_row["array_size"]
        assert min_cost >= 0

        array_dim /= zoom_factor
        battery_dim /= zoom_factor
        array_density /= zoom_factor
        battery_density /= zoom_factor

    assert min_cost is not None
    assert best_row is not None

    # If the optimal sizes are close to the search boundary, we should increase our
    # space to increase the likelihood that we're getting the global optimum.
    if (
        best_row["battery_size"] >= START_BATTERY_DIM
        or best_row["array_size"] >= START_ARRAY_DIM
    ):
        print(
            "Warning: optimal power configuration approached search boundary, you may want to consider tweaking search dimensions."
        )

    return {
        OptimizeColumn.SOLAR_COST_MW: solar_cost,
        OptimizeColumn.BATTERY_COST_MWH: battery_cost,
        OptimizeColumn.LOAD_COST_MW: load_cost,
        OptimizeColumn.ARRAYSIZE_MW: array_size,
        OptimizeColumn.BATTERY_SIZE_MWH: battery_size,
        OptimizeColumn.LOAD_SIZE_MW: load,
        OptimizeColumn.ARRAY_COST: solar_cost * array_size,
        OptimizeColumn.BATTERY_COST: battery_cost * battery_size,
        OptimizeColumn.LOAD_COST_NORMALIZED: load_cost,
        OptimizeColumn.TOTAL_POWER_SYSTEM_COST: solar_cost * array_size
        + battery_cost * battery_size,
        OptimizeColumn.TOTAL_SYSTEM_COST: solar_cost * array_size
        + battery_cost * battery_size
        + load_cost,
        OptimizeColumn.TOTAL_SYSTEM_COST_PER_UTILIZATION: min_cost,
        OptimizeColumn.BATTERY_SIZE_RELATIVE: best_row["capacity"],
        OptimizeColumn.LOAD_SIZE_RELATIVE: best_row["load"],
        OptimizeColumn.ANNUAL_BATTERY_UTILIZATION: best_row["uptime"],
        OptimizeColumn.ANNUAL_LOAD_UTILIZATION: best_row["utilization"],
    }


def compute_optimal_power_across_loads(
    solar_cost: float,
    battery_cost: float,
    load_costs: list[float],
    load: float,
    sol: np.ndarray,
) -> pd.DataFrame:
    results = []
    for i, load_cost in enumerate(load_costs):
        print(
            f"Optimizing load cost: {i}/{len(load_costs)}: {round(load_cost, 2)}",
            end="\r",
        )
        results.append(
            find_minimum_system_cost_parallel(
                solar_cost, battery_cost, load_cost, load, sol
            )
        )
    print(" " * 100, end="\r")
    return pd.DataFrame(results)


# Re-implementation of the reference mathematica notebook
# using gradient descent for optimization


def cost_and_elasticity(
    solar_cost: float,
    battery_cost: float,
    load_cost: float,
    battery_size: float,
    array_size: float,
    sol: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Computes cost and elasticity (partial derivatives) for battery and array sizes.
    """
    # Base cost
    load = 1.0
    perturbed_battery_size = 1.01 * battery_size + 0.01
    perturbed_array_size = 1.01 * array_size + 0.01
    costs = uptime_with_battery_with_inputs(
        load,
        np.array([battery_size, perturbed_battery_size]),
        np.array([array_size, perturbed_array_size]),
        sol,
    )

    costs["cost"] = all_in_system_cost(
        solar_cost,
        battery_cost,
        load_cost,
        costs["capacity"].to_numpy(),
        costs["load"].to_numpy(),
        costs["utilization"].to_numpy(),
    )
    cost = costs.loc[
        (costs["battery_size"] == battery_size) & (costs["array_size"] == array_size)
    ]["cost"].iloc[0]
    cost_battery = costs.loc[
        (costs["battery_size"] == perturbed_battery_size)
        & (costs["array_size"] == array_size)
    ]["cost"].iloc[0]
    cost_array = costs.loc[
        (costs["battery_size"] == battery_size)
        & (costs["array_size"] == perturbed_array_size)
    ]["cost"].iloc[0]

    # Calculate elasticities (relative change in cost)
    battery_elasticity = (cost - cost_battery) / cost if cost != 0 else 0
    array_elasticity = (cost - cost_array) / cost if cost != 0 else 0

    return cost, cost_battery, cost_array, battery_elasticity, array_elasticity


def find_minimum_system_cost_gradient(
    solar_cost: float,
    battery_cost: float,
    load_cost: float,
    sol: np.ndarray,
) -> dict[str, Any]:
    """
    Gradient-based optimization version that mirrors the Mathematica implementation.
    Uses elasticity to guide the search direction.
    """
    # Initial guesses based on load cost
    battery_initial = min(10.0, 10.0 * load_cost / 5_000_000)
    array_initial = min(10.0, 1.0 + 9.0 * load_cost / 5_000_000)

    # Amplitude for step size
    amp = 100 + 700.0 * (load_cost / 5_000_000) ** 1

    # Special case adjustments (matching Mathematica logic)
    if 700_000 < load_cost < 1_300_000:
        amp *= 3
    if load_cost > 80_000_000:
        amp *= 0.5

    steps = 10
    cost_min = 1e10
    battery_min = battery_initial
    array_min = array_initial

    battery_current = battery_initial
    array_current = array_initial

    for i in range(steps):
        # Calculate cost and elasticity
        cost, cost_b, cost_a, battery_elast, array_elast = cost_and_elasticity(
            solar_cost, battery_cost, load_cost, battery_current, array_current, sol
        )

        # Update minimum if we found a better solution
        if cost < cost_min:
            array_min = array_current
            battery_min = battery_current
            cost_min = cost

        # Update positions using elasticity as gradient
        # Random factor between 0.1 and 1.0 for exploration
        random_factor_b = np.random.uniform(0.1, 1.0)
        random_factor_a = np.random.uniform(0.1, 1.0)

        battery_current = max(
            0.0, battery_current + amp * random_factor_b * battery_elast
        )
        array_current = max(0.01, array_current + amp * random_factor_a * array_elast)

    # Calculate final utilization for the optimal solution
    load = 1.0
    final_result = uptime_with_battery_with_inputs(
        load, np.array([battery_min]), np.array([array_min]), sol
    )
    final_uptime = final_result.iloc[0]["uptime"]
    final_utilization = final_result.iloc[0]["utilization"]

    # Return in the same format as the other optimization function
    return {
        OptimizeColumn.SOLAR_COST_MW: solar_cost,
        OptimizeColumn.BATTERY_COST_MWH: battery_cost,
        OptimizeColumn.LOAD_COST_MW: load_cost,
        OptimizeColumn.ARRAYSIZE_MW: array_min,
        OptimizeColumn.BATTERY_SIZE_MWH: battery_min,
        OptimizeColumn.LOAD_SIZE_MW: load,
        OptimizeColumn.ARRAY_COST: solar_cost * array_min,
        OptimizeColumn.BATTERY_COST: battery_cost * battery_min,
        OptimizeColumn.LOAD_COST_NORMALIZED: load_cost,
        OptimizeColumn.TOTAL_POWER_SYSTEM_COST: solar_cost * array_min
        + battery_cost * battery_min,
        OptimizeColumn.TOTAL_SYSTEM_COST: solar_cost * array_min
        + battery_cost * battery_min
        + load_cost,
        OptimizeColumn.TOTAL_SYSTEM_COST_PER_UTILIZATION: cost_min,
        OptimizeColumn.BATTERY_SIZE_RELATIVE: battery_min / array_min,
        OptimizeColumn.LOAD_SIZE_RELATIVE: load / array_min,
        OptimizeColumn.ANNUAL_BATTERY_UTILIZATION: final_uptime,
        OptimizeColumn.ANNUAL_LOAD_UTILIZATION: final_utilization,
    }


def compute_optimal_power_across_loads_gradient(
    solar_cost: float,
    battery_cost: float,
    load_costs: list[float],
    load: float,
    sol: np.ndarray,
) -> pd.DataFrame:
    """Version using the gradient-based optimization"""
    return pd.DataFrame(
        [
            find_minimum_system_cost_gradient(solar_cost, battery_cost, load_cost, sol)
            for load_cost in load_costs
        ]
    )


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


def plot_state_map(files_df: pd.DataFrame, state: State, output_filepath: str):
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

    m.save(output_filepath)
    print(f"Plot successfully saved to {output_filepath}")


# Plot optimization results


def plot_cost_by_utilization(df: pd.DataFrame, output_filepath: str):
    utilization = df[OptimizeColumn.ANNUAL_LOAD_UTILIZATION]
    solar_cost = df[OptimizeColumn.ARRAY_COST]
    battery_cost = df[OptimizeColumn.BATTERY_COST]
    load_cost = df[OptimizeColumn.LOAD_COST_NORMALIZED]
    power_cost = df[OptimizeColumn.TOTAL_POWER_SYSTEM_COST]
    total_cost = df[OptimizeColumn.TOTAL_SYSTEM_COST]
    plt.plot(utilization, solar_cost, label="Solar")
    plt.plot(utilization, battery_cost, label="Battery")
    plt.plot(utilization, load_cost, label="Load")
    plt.plot(utilization, power_cost, label="Power")
    plt.plot(utilization, total_cost, label="Total")

    plt.title("What contributes to cost as you seek higher utilization?")
    plt.xlabel("Utilization")
    plt.ylabel("Cost ($/MW)")
    plt.xlim([1e-3, 1])
    plt.ylim([1 * 10**4, 1 * 10**7])
    plt.legend()
    plt.savefig(output_filepath)
    plt.close()
    print(f"Plot successfully saved to {output_filepath}")


def plot_sub_cost_by_load_cost(df: pd.DataFrame, output_filepath: str):
    solar_cost = df[OptimizeColumn.ARRAY_COST]
    battery_cost = df[OptimizeColumn.BATTERY_COST]
    load_cost = df[OptimizeColumn.LOAD_COST_NORMALIZED]
    power_cost = df[OptimizeColumn.TOTAL_POWER_SYSTEM_COST]
    total_cost = df[OptimizeColumn.TOTAL_SYSTEM_COST]
    total_cost_per_util = df[OptimizeColumn.TOTAL_SYSTEM_COST_PER_UTILIZATION]
    plt.plot(load_cost, solar_cost, label="Solar")
    plt.plot(load_cost, battery_cost, label="Battery")
    plt.plot(load_cost, load_cost, label="Load")
    plt.plot(load_cost, power_cost, label="Power")
    plt.plot(load_cost, total_cost, label="Total")
    plt.plot(load_cost, total_cost_per_util, label="Total per utilization")

    plt.title("How to component costs change to cost with load cost?")
    plt.xlabel("Load capex ($/MW)")
    plt.ylabel("Cost ($/MW)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(output_filepath)
    plt.close()
    print(f"Plot successfully saved to {output_filepath}")


def plot_power_cost_per_energy_by_utilization(df: pd.DataFrame, output_filepath: str):
    utilization = df[OptimizeColumn.ANNUAL_LOAD_UTILIZATION]
    solar_cost = df[OptimizeColumn.ARRAY_COST]
    battery_cost = df[OptimizeColumn.BATTERY_COST]
    power_cost = df[OptimizeColumn.TOTAL_POWER_SYSTEM_COST]

    u_values = np.arange(0.08, utilization.iloc[0] + 0.001, 0.001)
    v_values = np.full(len(u_values), solar_cost.iloc[0]) / (10 * 8760 * u_values)

    plt.plot(utilization, solar_cost / (10 * 8760 * utilization), label="Solar")
    plt.plot(utilization, battery_cost / (10 * 8760 * utilization), label="Battery")
    plt.plot(
        utilization, power_cost / (10 * 8760 * utilization), label="Total Power System"
    )
    plt.plot(u_values, v_values, label="Under-utilized solar")

    plt.title("What is the cost of the power system across varying load costs?")
    plt.xlabel("Load utilization")
    plt.ylabel("$/MWh when used")
    plt.xlim([1e-3, 1])
    plt.legend()
    plt.savefig(output_filepath)
    plt.close()
    print(f"Plot successfully saved to {output_filepath}")


def plot_utilization_by_load_cost(df: pd.DataFrame, output_filepath: str):
    utilization = df[OptimizeColumn.ANNUAL_LOAD_UTILIZATION]
    load_cost = df[OptimizeColumn.LOAD_COST_NORMALIZED]
    plt.plot(load_cost, utilization)
    plt.title("What is the relationship with load cost and optimal utilization?")
    plt.xlabel("Load capex ($/MW)")
    plt.ylabel("Optimal utilization")
    plt.xscale("log")
    plt.savefig(output_filepath)
    plt.close()
    print(f"Plot successfully saved to {output_filepath}")


def plot_power_cost_per_energy_by_load_cost_locations(
    dfs: list[tuple[str, pd.DataFrame]], output_filepath: str
):
    for location, df in dfs:
        utilization = df[OptimizeColumn.ANNUAL_LOAD_UTILIZATION]
        load_cost = df[OptimizeColumn.LOAD_COST_NORMALIZED]
        power_cost = df[OptimizeColumn.TOTAL_POWER_SYSTEM_COST]
        plt.scatter(load_cost, power_cost / (10 * 8760 * utilization), label=location)

    plt.title(
        "How much does the power system cost per energy across different load costs + locations?"
    )
    plt.xlabel("Load capex ($/MW)")
    plt.ylabel("Power system capex ($/MWh-load over 10 years)")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(output_filepath)
    plt.close()
    print(f"Plot successfully saved to {output_filepath}")


def plot_power_cost_per_energy_by_load_cost(
    location: str, output_df: pd.DataFrame, output_filepath: str
):
    plot_power_cost_per_energy_by_load_cost_locations(
        [(location, output_df)], output_filepath
    )


# Plotting TODO:
# - File meta data
#   - Plot histogram of capacities for a state
# - Plot utilization for each day over a year
# - Plot power generation over a day, overlaying all days onto one graph

# All


def mean_plant_for_state(directory: str, state: State) -> pd.DataFrame:
    state_file_df = create_state_files_df(directory, [state])
    state_files_5min = state_file_df.loc[
        state_file_df[DatasetColumn.TIME_INTERVAL_MIN] == 5
    ]
    plant_count = len(state_files_5min)

    dfs = []
    idx = 0
    for _, row in state_files_5min.iterrows():
        sol_metadata = row.to_dict()
        filename = filename_from_dict(row.to_dict())
        print(f"Reading file {idx}/{plant_count}: {filename}", end="\r")
        sol_df = read_plant_csv(directory, state, filename)
        sol_df[PowerColumn.POWER_MW] /= sol_metadata[DatasetColumn.CAPACITY_MW]
        dfs.append(sol_df)
        idx += 1
    print(" " * 100, end="\r")
    combined_df = pd.concat(dfs, ignore_index=True)
    return (
        combined_df.groupby(PowerColumn.LOCAL_TIME)[PowerColumn.POWER_MW]
        .mean()
        .reset_index()
    )


def do_all(
    directory: str,
    state: State,
    solar_cost: float,
    battery_cost: float,
    output_directory: str,
) -> pd.DataFrame:
    print(f"Processing {state.full_name()} ...")
    download_state_solar_data(directory, state)

    # Get average sol for each
    sol_df = mean_plant_for_state(directory, state)

    # Optimize on each
    optimize_df = compute_optimal_power_across_loads(
        solar_cost=solar_cost,
        battery_cost=battery_cost,
        load_costs=DEFAULT_LOAD_COSTS,
        load=1.0,
        sol=sol_df[PowerColumn.POWER_MW].to_numpy(),
    )

    output_filepath = f"{output_directory}/mean_power_optimal.csv"
    optimize_df.to_csv(output_filepath)
    print(f"Optimization complete. Results saved to `{output_filepath}`")

    # All plots
    files_df = create_state_files_df(directory, [state])
    plot_state_map(files_df, state, f"{output_directory}/map.html")
    plot_cost_by_utilization(optimize_df, f"{output_directory}/cost_by_util.png")
    plot_sub_cost_by_load_cost(
        optimize_df, f"{output_directory}/sub_cost_by_load_cost.png"
    )
    plot_power_cost_per_energy_by_utilization(
        optimize_df, f"{output_directory}/power_cost_per_energy_by_util.png"
    )
    plot_utilization_by_load_cost(optimize_df, f"{output_directory}/util_by_cost.png")
    plot_power_cost_per_energy_by_load_cost(
        state.full_name(),
        optimize_df,
        f"{output_directory}/power_cost_per_energy_by_load_cost.png",
    )

    return optimize_df


# Main CLI


class Command(StringEnum):
    DOWNLOAD = "download"
    OPTIMIZE = "optimize"
    PLOT = "plot"
    ALL = "all"


class PlotKind(StringEnum):
    MAP = "map"
    COST_BY_UTIL = "cost-by-util"
    SUB_COST_BY_LOAD_COST = "sub-cost-by-load-cost"
    POWER_COST_PER_ENERGY_BY_UTIL = "power-cost-by-util"
    UTIL_BY_COST = "util-by-cost"
    POWER_COST_PER_ENERGY_BY_LOAD_COST = "power-cost-by-load-cost"


DEFAULT_SOLAR_COST = 200_000  # $/MW
DEFAULT_BATTERY_COST = 200_000  # $/MWh
DEFAULT_LOAD_COSTS = list(10_000 * 10 ** np.arange(0, 0.2, 0.1))  # $/MW

DEFAULT_DATA_DIRECTORY = "data"
DEFAULT_OUTPUT_DIRECTORY = "output"
DEFAULT_OUTPUT_CSV = f"{DEFAULT_OUTPUT_DIRECTORY}/output.csv"
DEFAULT_OUTPUT_PLOT = f"{DEFAULT_OUTPUT_DIRECTORY}/output.png"


def main():
    parser = argparse.ArgumentParser(
        description="A tool to model cost and utilization of solar power systems."
    )
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # Download
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

    # Optimize
    optimize_parser = subparsers.add_parser(
        Command.OPTIMIZE,
        help="Produce optimal array and battery sizes for a range of load costs",
    )
    optimize_parser.add_argument(
        "--directory",
        default=DEFAULT_DATA_DIRECTORY,
        help=f"Directory where state data to is located. Defaults to `{DEFAULT_DATA_DIRECTORY}`",
    )
    optimize_parser.add_argument(
        "--state", required=True, help="Which state to plot a map of solar plants."
    )
    optimize_parser.add_argument(
        "--file",
        required=True,
        help="The filename for the plant's power generation dataset.",
    )
    optimize_parser.add_argument(
        "--solar-cost",
        type=float,
        default=DEFAULT_SOLAR_COST,
        help=f"$/MW used for solar cost in the optimization. Defaults to `200,000`",
    )
    optimize_parser.add_argument(
        "--battery-cost",
        type=float,
        default=DEFAULT_BATTERY_COST,
        help=f"$/MWh used for battery cost in the optimization. Defaults to `200,000`",
    )
    optimize_parser.add_argument(
        "--output-directory",
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Directory to output results to. Defaults to `{DEFAULT_OUTPUT_DIRECTORY}`",
    )
    optimize_parser.add_argument(
        "--reference",
        action="store_true",
        help="Optimize via reference re-implementation, i.e. gradient descent.",
    )

    # Plot
    plot_parser = subparsers.add_parser(
        Command.PLOT, help="Visualize preset plots for datasets."
    )
    plot_subparsers = plot_parser.add_subparsers(title="plot kinds", dest="plot_kind")

    # Plot - map
    plot_map_parser = plot_subparsers.add_parser(
        PlotKind.MAP, help="Plot solar plants for a state geographically."
    )
    plot_map_parser.add_argument(
        "--state", required=True, help="Which state to plot a map of solar plants."
    )
    plot_map_parser.add_argument(
        "--directory",
        default=DEFAULT_DATA_DIRECTORY,
        help=f"Used in `map` plot kind. Directory data was download to. Defaults to `{DEFAULT_DATA_DIRECTORY}`",  # Expects data directory to be structured as this tool downloads data
    )
    plot_map_parser.add_argument(
        "--output-directory",
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Directory to output results to. Defaults to `{DEFAULT_OUTPUT_DIRECTORY}`",
    )

    # Plot - cost by util
    plot_cost_by_util_parser = plot_subparsers.add_parser(
        PlotKind.COST_BY_UTIL,
        help="Plot cost of system components by optimal utilization.",
    )
    plot_cost_by_util_parser.add_argument(
        "--input", required=True, help="A CSV file produced by the `optimize` command."
    )
    plot_cost_by_util_parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PLOT, help="Where to save the plot."
    )

    # Plot - sub cost by load cost
    plot_sub_cost_by_load_cost_parser = plot_subparsers.add_parser(
        PlotKind.SUB_COST_BY_LOAD_COST,
        help="Plot cost of system components by load cost.",
    )
    plot_sub_cost_by_load_cost_parser.add_argument(
        "--input", required=True, help="A CSV file produced by the `optimize` command."
    )
    plot_sub_cost_by_load_cost_parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PLOT, help="Where to save the plot."
    )

    # Plot - power cost by util
    plot_power_cost_by_util_parser = plot_subparsers.add_parser(
        PlotKind.POWER_COST_PER_ENERGY_BY_UTIL,
        help="Plot cost of power system components per energy by optimal utilization.",
    )
    plot_power_cost_by_util_parser.add_argument(
        "--input", required=True, help="A CSV file produced by the `optimize` command."
    )
    plot_power_cost_by_util_parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PLOT, help="Where to save the plot."
    )

    # Plot - util by cost
    plot_util_by_cost_parser = plot_subparsers.add_parser(
        PlotKind.UTIL_BY_COST, help="Plot optimal utilization by load cost."
    )
    plot_util_by_cost_parser.add_argument(
        "--input", required=True, help="A CSV file produced by the `optimize` command."
    )
    plot_util_by_cost_parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PLOT, help="Where to save the plot."
    )

    # Plot - power cost per energy by load cost
    plot_power_cost_per_energy_by_load_cost_parser = plot_subparsers.add_parser(
        PlotKind.POWER_COST_PER_ENERGY_BY_LOAD_COST,
        help="Plot power cost per energy usage by load cost.",
    )
    plot_power_cost_per_energy_by_load_cost_parser.add_argument(
        "--state",
        required=True,
        help="Which state is being plotted. Used to label the plot.",
    )
    plot_power_cost_per_energy_by_load_cost_parser.add_argument(
        "--input", required=True, help="A CSV file produced by the `optimize` command."
    )
    plot_power_cost_per_energy_by_load_cost_parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PLOT, help="Where to save the plot."
    )

    # TODO: everything (optimize for a state's average plant, not literally every single one)
    all_parser = subparsers.add_parser(
        Command.ALL,
        help="Download, optimize, and plot average plants across several default states",
    )
    all_parser.add_argument(
        "--directory",
        default=DEFAULT_DATA_DIRECTORY,
        help=f"Directory where state data to is located. Defaults to `{DEFAULT_DATA_DIRECTORY}`",
    )
    all_parser.add_argument(
        "--states",
        nargs="*",
        type=str,
        default=[
            State.ARIZONA,
            State.CALIFORNIA,
            State.MAINE,
            State.TEXAS,
            State.WASHINGTON,
        ],
        help="States to process.",
    )
    all_parser.add_argument(
        "--solar-cost",
        type=float,
        default=DEFAULT_SOLAR_COST,
        help=f"$/MW used for solar cost in the optimization. Defaults to `200,000`",
    )
    all_parser.add_argument(
        "--battery-cost",
        type=float,
        default=DEFAULT_BATTERY_COST,
        help=f"$/MWh used for battery cost in the optimization. Defaults to `200,000`",
    )
    all_parser.add_argument(
        "--output-directory",
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Directory to output results to. Defaults to `{DEFAULT_OUTPUT_DIRECTORY}`",
    )

    args = parser.parse_args()

    if args.command == Command.DOWNLOAD:
        if args.all == (args.state != ""):
            print("Error: Please specify either --state or --all")
            return
        if not os.path.exists(args.directory) or not os.path.isdir(args.directory):
            print(
                f"Data directory '{args.directory}' either doesn't exist or is not a directory"
            )
            return
        if args.state != "" and not State.valid(args.state):
            print(
                f"Error: {args.state} is not a valid state. Available states: {', '.join(State.all())}"
            )
            return
        states = []
        if args.all:
            states = [state for state in State]
        else:
            states = [State.from_str(args.state)]
        for state in states:
            download_state_solar_data(args.directory, state)
    elif args.command == Command.OPTIMIZE:
        if not State.valid(args.state):
            print(
                f"Error: {args.state} is not a valid state. Available states: {', '.join(State.all())}"
            )
            return
        if not os.path.exists(args.directory) or not os.path.isdir(args.directory):
            print(
                f"Data directory '{args.directory}' either doesn't exist or is not a directory"
            )
            return
        if os.path.exists(args.output_directory) and not os.path.isdir(
            args.output_directory
        ):
            print(
                f"Output directory '{args.output_directory}' exists but is not a directory"
            )
            return

        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)
            print(f"Created directory: {args.output_directory}")

        state = State.from_str(args.state)
        sol_metadata = metadata_from_filename(state, args.file)
        sol_df = read_plant_csv(args.directory, state, args.file)
        sol = (
            sol_df[PowerColumn.POWER_MW] / sol_metadata[DatasetColumn.CAPACITY_MW]
        ).to_numpy()

        optimize_df = pd.DataFrame()
        if args.reference:
            optimize_df = compute_optimal_power_across_loads_gradient(
                solar_cost=args.solar_cost,
                battery_cost=args.battery_cost,
                load_costs=DEFAULT_LOAD_COSTS,
                load=1.0,
                sol=sol,
            )
        else:
            optimize_df = compute_optimal_power_across_loads(
                solar_cost=args.solar_cost,
                battery_cost=args.battery_cost,
                load_costs=DEFAULT_LOAD_COSTS,
                load=1.0,
                sol=sol,
            )
        output_filepath = f"{DEFAULT_OUTPUT_DIRECTORY}/output.csv"
        optimize_df.to_csv(output_filepath)
        print(f"Optimization complete. Results saved to `{output_filepath}`")
    elif args.command == Command.PLOT:
        if args.plot_kind == PlotKind.MAP:
            if not State.valid(args.state):
                print(
                    f"Error: {args.state} is not a valid state. Available states: {', '.join(State.all())}"
                )
                return
            if os.path.exists(args.output_directory) and not os.path.isdir(
                args.output_directory
            ):
                print(
                    f"Output directory '{args.output_directory}' exists but is not a directory"
                )
                return
            state = State.from_str(args.state)
            files_df = create_state_files_df(args.directory, [state for state in State])
            state_dir = f"{args.output_directory}/{state}"
            if not os.path.exists(state_dir):
                os.makedirs(state_dir)
                print(f"Created directory: {state_dir}")
            plot_state_map(files_df, state, f"{state_dir}/map.html")
        elif args.plot_kind == PlotKind.COST_BY_UTIL:
            if not os.path.isfile(args.input) or not args.input.endswith(".csv"):
                print(f"Error: {args.input} is not a valid input file.")
            else:
                df = pd.read_csv(args.input)
                plot_cost_by_utilization(df, args.output)
        elif args.plot_kind == PlotKind.SUB_COST_BY_LOAD_COST:
            if not os.path.isfile(args.input) or not args.input.endswith(".csv"):
                print(f"Error: {args.input} is not a valid input file.")
            else:
                df = pd.read_csv(args.input)
                plot_sub_cost_by_load_cost(df, args.output)
        elif args.plot_kind == PlotKind.POWER_COST_PER_ENERGY_BY_UTIL:
            if not os.path.isfile(args.input) or not args.input.endswith(".csv"):
                print(f"Error: {args.input} is not a valid input file.")
            else:
                df = pd.read_csv(args.input)
                plot_power_cost_per_energy_by_utilization(df, args.output)
        elif args.plot_kind == PlotKind.UTIL_BY_COST:
            if not os.path.isfile(args.input) or not args.input.endswith(".csv"):
                print(f"Error: {args.input} is not a valid input file.")
            else:
                df = pd.read_csv(args.input)
                plot_utilization_by_load_cost(df, args.output)
        elif args.plot_kind == PlotKind.POWER_COST_PER_ENERGY_BY_LOAD_COST:
            if not os.path.isfile(args.input) or not args.input.endswith(".csv"):
                print(f"Error: {args.input} is not a valid input file.")
            elif not State.valid(args.state):
                print(
                    f"Error: {args.state} is not a valid state. Available states: {', '.join(State.all())}"
                )
            else:
                df = pd.read_csv(args.input)
                plot_power_cost_per_energy_by_load_cost(
                    State.from_str(args.state).full_name(), df, args.output
                )
        else:
            plot_parser.print_help()
    elif args.command == Command.ALL:
        for state in args.states:
            if not State.valid(state):
                print(
                    f"Error: {state} is not a valid state. Available states: {', '.join(State.all())}"
                )
                return
        if not os.path.exists(args.directory) or not os.path.isdir(args.directory):
            print(
                f"Data directory '{args.directory}' either doesn't exist or is not a directory"
            )
            return
        if os.path.exists(args.output_directory) and not os.path.isdir(
            args.output_directory
        ):
            print(
                f"Output directory '{args.output_directory}' exists but is not a directory"
            )
            return

        states = [State.from_str(state) for state in args.states]
        optimize_dfs = []
        for state in states:
            state_dir = f"{args.output_directory}/{state}"
            if not os.path.exists(state_dir):
                os.makedirs(state_dir)
                print(f"Created directory: {state_dir}")
            optimize_df = do_all(
                args.directory, state, args.solar_cost, args.battery_cost, state_dir
            )
            optimize_dfs.append(optimize_df)
        plot_power_cost_per_energy_by_load_cost_locations(
            list(zip([state.full_name() for state in states], optimize_dfs)),
            f"{args.output_directory}/states_power_cost_per_energy_by_load_cost.png",
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
