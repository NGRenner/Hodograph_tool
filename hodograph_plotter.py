import os
import requests
import re
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import metpy.calc as mpcalc
from metpy.units import units
import sharppy.sharptab.profile as profile
from metpy.plots import Hodograph
import cfgrib
from datetime import datetime, timedelta

NOMADS_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/"

def get_latest_rap_file():
    now = datetime.utcnow()
    date_str = now.strftime('%Y%m%d')
    rap_dir = f"rap.{date_str}/"
    url = f"{NOMADS_URL}{rap_dir}"

    response = requests.get(url)
    if response.status_code != 200:
        now -= timedelta(days=1)
        date_str = now.strftime('%Y%m%d')
        rap_dir = f"rap.{date_str}/"
        url = f"{NOMADS_URL}{rap_dir}"
        response = requests.get(url)
        if response.status_code != 200:
            print("Failed to access RAP data.")
            return None

    files = re.findall(r'rap\.t(\d{2})z\.awp130pgrbf(\d{2})\.grib2', response.text)
    if not files:
        print("No RAP files found!")
        return None

    files = sorted(files, key=lambda x: (int(x[0]), int(x[1])), reverse=True)
    latest_cycle, latest_fxx = files[0]
    latest_file = f"rap.t{latest_cycle}z.awp130pgrbf{latest_fxx}.grib2"
    latest_url = f"{url}{latest_file}"
    return latest_url, latest_file

def download_rap_file():
    result = get_latest_rap_file()
    if not result:
        return None
    latest_url, latest_file = result

    if os.path.exists(latest_file):
        print(f"✅ File already exists: {latest_file}")
        return latest_file

    print(f"Downloading: {latest_url}")
    response = requests.get(latest_url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(latest_file, 'wb') as f, tqdm(
            desc="Downloading RAP File", total=total_size, unit='B', unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
        return latest_file
    else:
        print("❌ Failed to download RAP file.")
        return None

def replace_nan_with_missing(arr):
    return [-9999 if np.isnan(x) else x for x in arr]

def plot_hodograph_sharppy(data_file, lat, lon):
    datasets = cfgrib.open_datasets(
        data_file,
        backend_kwargs={'decode_timedelta': True}
    )
    ds = datasets[19]  # isobaricInhPa dataset

    lat_values = ds["latitude"].values
    lon_values = ds["longitude"].values

    lat_idx = np.clip(np.abs(lat_values - lat).argmin(), 0, lat_values.shape[0] - 1)
    lon_idx = np.clip(np.abs(lon_values - lon).argmin(), 0, lon_values.shape[1] - 1)

    pressure = ds["isobaricInhPa"].values
    temp_k = ds["t"].isel(y=lat_idx, x=lon_idx).values
    rh = ds["r"].isel(y=lat_idx, x=lon_idx).values * units.percent
    u = ds["u"].isel(y=lat_idx, x=lon_idx).values
    v = ds["v"].isel(y=lat_idx, x=lon_idx).values

    # Optional height (Geopotential meters → meters)
    try:
        height = ds["gh"].isel(y=lat_idx, x=lon_idx).values
        height = replace_nan_with_missing(height.tolist())
    except Exception:
        height = None

    temp = (temp_k - 273.15) * units.degC
    dewpoint = mpcalc.dewpoint_from_relative_humidity(temp, rh)

    # Convert winds to knots
    u_knots = (u * units.meter / units.second).to('knots').magnitude
    v_knots = (v * units.meter / units.second).to('knots').magnitude

    # Prepare all arrays
    pressure = replace_nan_with_missing(pressure.tolist())
    temp = replace_nan_with_missing(temp.magnitude.tolist())
    dewpoint = replace_nan_with_missing(dewpoint.magnitude.tolist())
    u_knots = replace_nan_with_missing(u_knots.tolist())
    v_knots = replace_nan_with_missing(v_knots.tolist())

    print("\n🔹 Debugging SHARPpy Input Data:")
    print(f"Pressure [{len(pressure)}] hPa: {pressure[:5]} ...")
    print(f"Temp [{len(temp)}] °C: {temp[:5]} ...")
    print(f"Dewpoint [{len(dewpoint)}] °C: {dewpoint[:5]} ...")
    print(f"U (kt) [{len(u_knots)}]: {u_knots[:5]} ...")
    print(f"V (kt) [{len(v_knots)}]: {v_knots[:5]} ...")
    if height:
        print(f"Height (gpm): {height[:5]} ...")

    # Ensure descending pressure
    if not np.all(np.diff(pressure) < 0):
        print("⚠️ Sorting data by descending pressure...")
        sort_idx = np.argsort(pressure)[::-1]
        pressure = [pressure[i] for i in sort_idx]
        temp = [temp[i] for i in sort_idx]
        dewpoint = [dewpoint[i] for i in sort_idx]
        u_knots = [u_knots[i] for i in sort_idx]
        v_knots = [v_knots[i] for i in sort_idx]
        if height:
            height = [height[i] for i in sort_idx]

    try:
        prof = profile.create_profile(
            profile="RAP",
            pres=pressure,
            hght=height,
            tmpc=temp,
            dwpc=dewpoint,
            u=u_knots,
            v=v_knots,
            missing=-9999,
            strictQC=True
        )
        if prof is None:
            raise ValueError("SHARPpy returned None.")
    except Exception as e:
        print(f"❌ SHARPpy profile creation failed: {e}")
        return

    print("✅ SHARPpy profile successfully created!")

    fig, ax = plt.subplots(figsize=(6, 6))
    h = Hodograph(ax, component_range=40)
    h.add_grid(increment=10)

    if hasattr(prof, "srwind"):
        h.plot(prof.srwind[0], prof.srwind[1], color='red', linestyle="dashed", label="Storm-Relative Wind")

    h.plot(prof.u, prof.v, color='blue', marker='o', label="Observed Wind")
    plt.legend()
    plt.title(f"Hodograph for {lat}, {lon} (SHARPpy)")
    plt.show()

if __name__ == "__main__":
    data_file = download_rap_file()
    if data_file:
        plot_hodograph_sharppy(data_file, lat=35.4676, lon=-97.5164)
