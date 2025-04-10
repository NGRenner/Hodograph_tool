#!/usr/bin/env python3
import os
import re
import requests
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.interpolate import griddata, interp1d
import pandas as pd
import imageio
import pygrib

# MetPy imports
from metpy.calc import (
    dewpoint_from_relative_humidity,
    parcel_profile,
    lcl,
    cape_cin
)
from metpy.plots import SkewT
from metpy.units import units

# Model‑specific information for NOMADS
model_info = {
    "rap": {
        "base": "http://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/",
        "folder": "rap",
        "regex": r'rap\.t(\d{2})z\.awp130pgrbf(\d{2})\.grib2(?<!\.idx)'
    },
    "nam": {
        "base": "http://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/",
        "folder": "nam",
        # Updated NAM regex to match files like: nam.tHHz.awphysBB.tm00.grib2
        "regex": r'nam\.t(\d{2})z\.awphys(\d{2})\.tm00\.grib2'
    },
    "gfs": {
        "base": "http://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/",
        "folder": "gfs",
        "subfolder": "atmos/",
        "regex": r'gfs\.t(\d{2})z\.pgrb2full\.0p50\.f(\d{3})'
    }
}

# RAP-specific function (also used for NAM in the original script)
def get_available_model_cycles(model):
    """For RAP: returns (base_url, available, cycles, date_str)."""
    info = model_info[model]
    now = datetime.utcnow()
    for attempt in range(2):
        date_str = now.strftime('%Y%m%d')
        folder = f"{info['folder']}.{date_str}/"
        base_url = info["base"] + folder
        response = requests.get(base_url)
        if response.status_code != 200:
            now -= timedelta(days=1)
            continue
        files = re.findall(info["regex"], response.text)
        if not files:
            now -= timedelta(days=1)
            continue
        available = [(cycle, fxx, f"{info['folder']}.t{cycle}z.awp130pgrbf{fxx}.grib2") for (cycle, fxx) in files]
        available = list(set(available))
        cycles = sorted(set([t[0] for t in available]), reverse=True)
        return base_url, available, cycles, date_str
    print("⚠️ No valid GRIB2 files found for", model)
    return None, None, None, None

# NAM‑specific functions
def get_available_nam_dates():
    """Returns available NAM dates from the base directory."""
    base_url = model_info["nam"]["base"]
    response = requests.get(base_url)
    if response.status_code != 200:
        print("Could not access NAM base directory.")
        return None, None
    dates = re.findall(r'nam\.(\d{8})', response.text)
    dates = sorted(set(dates), reverse=True)
    return base_url, dates

def get_available_nam_cycle(base_url, chosen_date):
    """For a given NAM date, returns available cycles."""
    date_folder = f"nam.{chosen_date}/"
    full_url = base_url + date_folder
    response = requests.get(full_url)
    if response.status_code != 200:
        print("Could not access NAM date folder:", full_url)
        return None, None
    matches = re.findall(model_info["nam"]["regex"], response.text)
    if not matches:
        print("No NAM files found in the directory.")
        return None, None
    cycles = sorted(set([m[0] for m in matches]), reverse=True)
    return full_url, cycles

def get_available_nam_valid_times(full_url, chosen_cycle, chosen_date):
    """For a given NAM cycle, returns the list of valid times."""
    info = model_info["nam"]
    response = requests.get(full_url)
    if response.status_code != 200:
        print("Could not access NAM date folder:", full_url)
        return None, None
    files = re.findall(info["regex"], response.text)
    if not files:
        print("No valid NAM files found in the directory.")
        return None, None
    available_files = []
    for (cycle, fxx) in files:
        fname = f"nam.t{cycle}z.awphys{fxx}.tm00.grib2"
        available_files.append((cycle, fxx, fname))
    # Filter for the chosen cycle
    available_files = [item for item in available_files if item[0] == chosen_cycle]
    if not available_files:
        print("No valid NAM files found for cycle", chosen_cycle)
        return None, None
    available_files = sorted(list(set(available_files)), key=lambda t: int(t[1]))
    base_date = datetime.strptime(chosen_date, "%Y%m%d")
    run_time = base_date + timedelta(hours=int(chosen_cycle))
    valid_list = []
    for t in available_files:
        fxx = t[1]
        valid_time = run_time + timedelta(hours=int(fxx))
        valid_list.append((valid_time, t[2], fxx))
    return full_url, valid_list

# GFS‑specific functions
def get_available_gfs_dates():
    base_url = model_info["gfs"]["base"]
    response = requests.get(base_url)
    if response.status_code != 200:
        print("Could not access GFS base directory.")
        return None, None
    dates = re.findall(r'gfs\.(\d{8})', response.text)
    dates = sorted(set(dates), reverse=True)
    return base_url, dates

def get_available_gfs_cycle(base_url, chosen_date):
    date_folder = f"gfs.{chosen_date}/"
    full_url = base_url + date_folder
    response = requests.get(full_url)
    if response.status_code != 200:
        print("Could not access GFS date folder:", full_url)
        return None, None
    cycles = re.findall(r'>(\d{2})/', response.text)
    cycles = sorted(set(cycles), reverse=True)
    return full_url, cycles

def get_available_gfs_valid_times(base_url_date, chosen_cycle, chosen_date):
    info = model_info["gfs"]
    final_base_url = base_url_date + f"{chosen_cycle}/" + info["subfolder"]
    response = requests.get(final_base_url)
    if response.status_code != 200:
        print("Could not access GFS subfolder for cycle", chosen_cycle)
        return None, None
    files = re.findall(info["regex"], response.text)
    if not files:
        print("No valid GFS files found in subfolder for cycle", chosen_cycle)
        return None, None
    available_files = []
    for (cycle, fxx) in files:
        fname = f"gfs.t{cycle}z.pgrb2full.0p50.f{fxx}"
        available_files.append((cycle, fxx, fname))
    available_files = sorted(list(set(available_files)), key=lambda t: int(t[1]))
    base_date = datetime.strptime(chosen_date, "%Y%m%d")
    run_time = base_date + timedelta(hours=int(chosen_cycle))
    valid_list = []
    for t in available_files:
        fxx = t[1]
        valid_time = run_time + timedelta(hours=int(fxx))
        valid_list.append((valid_time, t[2], fxx))
    return final_base_url, valid_list

def download_model_file(file_name, base_url):
    """Downloads the file if not already present."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    if os.path.exists(file_path):
        print(f"✅ File already exists: {file_path}")
        return file_path
    url = f"{base_url}{file_name}"
    print(f"Downloading: {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as f, tqdm(
            desc="Downloading File", total=total_size, unit='B', unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"✅ Downloaded file: {file_path}")
        return file_path
    else:
        print("❌ Failed to download file.")
        return None

def get_elevation_pygrib(file_path, lat_target, lon_target):
    """
    Gets the surface elevation from a GRIB2 file.
    """
    grbs = pygrib.open(file_path)
    try:
        orog_msgs = grbs.select(shortName='orog', typeOfLevel='surface')
    except Exception:
        orog_msgs = []
    if len(orog_msgs) == 0:
        try:
            orog_msgs = grbs.select(shortName='hgt', typeOfLevel='surface')
        except Exception:
            orog_msgs = []
    if len(orog_msgs) == 0:
        print("Warning: No orography/hgt field found in the file. Defaulting elevation to 0 m.")
        return 0.0
    grb = orog_msgs[0]
    print("Using GRIB message for orography/hgt:")
    print(grb)
    lats, lons = grb.latlons()
    if lats.ndim == 1 or lons.ndim == 1:
        lats, lons = np.meshgrid(lats, lons, indexing='ij')
    elevations = grb.values
    if lons.min() >= 0 and lons.max() > 180 and lon_target < 0:
        lon_target = lon_target + 360
    dist = np.sqrt((lats - lat_target)**2 + (lons - lon_target)**2)
    min_index = np.unravel_index(np.argmin(dist), dist.shape)
    return elevations[min_index]

def plot_skewt(data_file, lat, lon, model, output_file=None):
    """
    Reads temperature, relative humidity, wind, and geopotential height
    from a GRIB2 file, interpolates to the given lat/lon, and produces a
    Skew-T plot with CAPE/CIN shading and parcel profile.
    """
    ds_t = xr.open_dataset(
        data_file, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 't'}},
        indexpath=None, decode_timedelta=False
    )
    ds_rh = xr.open_dataset(
        data_file, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'r'}},
        indexpath=None, decode_timedelta=False
    )
    ds_u = xr.open_dataset(
        data_file, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}},
        indexpath=None, decode_timedelta=False
    )
    ds_v = xr.open_dataset(
        data_file, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}},
        indexpath=None, decode_timedelta=False
    )
    ds_z = xr.open_dataset(
        data_file, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh'}},
        indexpath=None, decode_timedelta=False
    )

    # Extract arrays
    t = ds_t["t"]
    rh = ds_rh["r"]
    u = ds_u["u"]
    v = ds_v["v"]
    z = ds_z["gh"]
    pressure = ds_t["isobaricInhPa"]
    lat_isobaric = ds_t["latitude"].values
    lon_isobaric = ds_t["longitude"].values

    # Get surface elevation
    station_elev_m = get_elevation_pygrib(data_file, lat, lon)
    print(f"Surface elevation at {lat:.2f}°N, {lon:.2f}° is ~{station_elev_m:.1f} m")

    rt = pd.to_datetime(ds_t.time.values)
    vt = pd.to_datetime(ds_t.valid_time.values)
    if lon_isobaric.min() >= 0 and lon_isobaric.max() > 180 and lon < 0:
        lon = lon + 360

    if lat_isobaric.ndim == 1 and lon_isobaric.ndim == 1:
        lat_grid, lon_grid = np.meshgrid(lat_isobaric, lon_isobaric, indexing='ij')
    else:
        lat_grid, lon_grid = lat_isobaric, lon_isobaric

    levels = [1000,975,950,925,900,875,850,825,800,775,
              750,725,700,675,650,625,600,575,550,525,
              500,475,450,425,400,375,350,325,300,275,
              250,225,200,175,150,125,100]

    t_interp, rh_interp, u_interp, v_interp, z_interp, levels_used = [], [], [], [], [], []
    for lev in tqdm(levels, desc="Interpolating Data", unit="level"):
        if lev not in pressure.values:
            continue
        idx = np.abs(pressure.values - lev).argmin()
        t_level = t.isel(isobaricInhPa=idx).values
        rh_level = rh.isel(isobaricInhPa=idx).values
        u_level = u.isel(isobaricInhPa=idx).values
        v_level = v.isel(isobaricInhPa=idx).values
        z_level = z.isel(isobaricInhPa=idx).values

        t_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        t_level.ravel(), (lat, lon), method='nearest')
        rh_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                         rh_level.ravel(), (lat, lon), method='nearest')
        u_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        u_level.ravel(), (lat, lon), method='nearest')
        v_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        v_level.ravel(), (lat, lon), method='nearest')
        z_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        z_level.ravel(), (lat, lon), method='nearest')

        t_interp.append(t_pt)
        rh_interp.append(rh_pt)
        u_interp.append(u_pt)
        v_interp.append(v_pt)
        z_interp.append(z_pt)
        levels_used.append(lev)

    t_interp = np.array(t_interp)
    rh_interp = np.array(rh_interp)
    u_interp = np.array(u_interp)
    v_interp = np.array(v_interp)
    z_interp = np.array(z_interp)
    levels_used = np.array(levels_used)

    p_profile = levels_used * units.hPa
    if model in ['rap', 'nam']:
        z_m = (z_interp / 9.80665) * 10.0
    else:
        z_m = (z_interp / 9.80665) * 10.0

    z_agl_m = z_m - station_elev_m

    T_kelvin = t_interp * units.K
    RH_frac = (rh_interp / 100.0)
    Td_kelvin = dewpoint_from_relative_humidity(T_kelvin, RH_frac)
    u_profile = u_interp * units('m/s')
    v_profile = v_interp * units('m/s')

    sort_idx = np.argsort(p_profile.m)[::-1]
    p_sorted = p_profile[sort_idx]
    T_sorted = T_kelvin[sort_idx]
    Td_sorted = Td_kelvin[sort_idx]
    u_sorted = u_profile[sort_idx]
    v_sorted = v_profile[sort_idx]
    z_sorted = z_agl_m[sort_idx] * units.meter

    fig = plt.figure(figsize=(9,9))
    skew = SkewT(fig, rotation=45)
    skew.plot(p_sorted, T_sorted.to('degC'), color='red', linewidth=2, label='Temperature')
    skew.plot(p_sorted, Td_sorted.to('degC'), color='green', linewidth=2, label='Dewpoint')
    
    step = max(1, len(p_sorted)//15)
    skew.plot_barbs(p_sorted[::step], u_sorted[::step], v_sorted[::step])
    
    p_sfc = p_sorted[0]
    T_sfc = T_sorted[0]
    Td_sfc = Td_sorted[0]
    lcl_p, lcl_t = lcl(p_sfc, T_sfc, Td_sfc)
    lcl_height = np.interp(lcl_p.m, p_sorted.m[::-1], z_sorted.m[::-1])
    print(f"LCL: {lcl_p:.1f}, {lcl_t.to('degC'):.1f} => ~{lcl_height:.0f} m AGL")
    prof = parcel_profile(p_sorted, T_sfc, Td_sfc)
    skew.plot(p_sorted, prof.to('degC'), color='black', linestyle='--', linewidth=2, label='Parcel')
    
    cape, cin = cape_cin(p_sorted, T_sorted, Td_sorted, prof)
    skew.shade_cin(p_sorted, T_sorted.to('degC'), prof.to('degC'), color='blue', alpha=0.2)
    skew.shade_cape(p_sorted, T_sorted.to('degC'), prof.to('degC'), color='red', alpha=0.2)
    
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    
    text_str = (f"CAPE: {cape.to('joules/kilogram').magnitude:.0f} J/kg\n"
                f"CIN: {cin.to('joules/kilogram').magnitude:.0f} J/kg\n"
                f"LCL: {lcl_height:.0f} m AGL")
    skew.ax.text(0.02, 0.02, text_str, transform=skew.ax.transAxes,
                 fontsize=10, color='navy',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    fcst_hour = int((vt - rt).total_seconds() // 3600)
    metadata_text = (
        f"{model.upper()} {rt.strftime('%Y-%m-%d %HZ')}, F{fcst_hour:03d}\n"
        f"VALID: {vt.strftime('%a %Y-%m-%d %HZ')}\n"
        f"AT: {lat:.2f}°N, {lon:.2f}°{'W' if lon < 0 else 'E'}"
    )
    plt.title("Skew-T/Log-P Diagram", loc='left', fontsize=11)
    skew.ax.text(0.98, 1.01, metadata_text, transform=skew.ax.transAxes,
                 ha='right', va='bottom',
                 bbox=dict(facecolor='black', alpha=0.8, edgecolor='none'),
                 color='white', fontsize=9)
    
    plt.legend(loc='best', fontsize=9)
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    model = input("Enter model (rap, nam, or gfs): ").strip().lower()
    if model not in model_info:
        print("Invalid model selection. Exiting.")
        return

    if model == "rap":
        base_url, available, cycles, date_str = get_available_model_cycles(model)
        if available is None:
            return
        print("Available RAP cycles:")
        for i, cycle in enumerate(cycles):
            print(f"{i}: {cycle}Z")
        cycle_index = int(input("Enter index for desired cycle: ").strip())
        chosen_cycle = cycles[cycle_index]
        files_for_chosen = sorted([t for t in available if t[0] == chosen_cycle], key=lambda t: int(t[1]))
        base_date = datetime.strptime(date_str, "%Y%m%d")
        run_time = base_date + timedelta(hours=int(chosen_cycle))
        valid_list = []
        for t in files_for_chosen:
            fxx = t[1]
            valid_time = run_time + timedelta(hours=int(fxx))
            valid_list.append((valid_time, t[2], fxx))
        download_base = base_url
    elif model == "nam":
        base_url, dates = get_available_nam_dates()
        if not dates:
            return
        print("Available NAM dates:")
        for i, d in enumerate(dates):
            print(f"{i}: nam.{d}")
        date_index = int(input("Enter index for desired date: ").strip())
        chosen_date = dates[date_index]
        full_url, cycles = get_available_nam_cycle(base_url, chosen_date)
        if not cycles:
            return
        print("Available NAM cycles:")
        for i, cycle in enumerate(cycles):
            print(f"{i}: {cycle}Z")
        cycle_index = int(input("Enter index for desired cycle: ").strip())
        chosen_cycle = cycles[cycle_index]
        full_url, valid_list = get_available_nam_valid_times(full_url, chosen_cycle, chosen_date)
        if valid_list is None:
            return
        download_base = full_url
    elif model == "gfs":
        base_url_gfs, gfs_dates = get_available_gfs_dates()
        if not gfs_dates:
            return
        print("Available GFS dates:")
        for i, d in enumerate(gfs_dates):
            print(f"{i}: gfs.{d}")
        date_index = int(input("Enter index for desired date: ").strip())
        chosen_date = gfs_dates[date_index]
        base_url_date, cycles = get_available_gfs_cycle(base_url_gfs, chosen_date)
        if not cycles:
            return
        print("Available run cycles:")
        for i, cycle in enumerate(cycles):
            print(f"{i}: {cycle}Z")
        cycle_index = int(input("Enter index for desired run cycle: ").strip())
        chosen_cycle = cycles[cycle_index]
        final_base_url, valid_list = get_available_gfs_valid_times(base_url_date, chosen_cycle, chosen_date)
        if valid_list is None:
            return
        download_base = final_base_url

    print("Available valid times:")
    for i, (vt, fname, fcst) in enumerate(valid_list):
        print(f"{i}: {vt.strftime('%Y-%m-%d %HZ')} - File: {fname}")
    start_idx = int(input("Enter start index for GIF: ").strip())
    end_idx = int(input("Enter end index for GIF: ").strip())
    if start_idx < 0 or end_idx >= len(valid_list) or start_idx > end_idx:
        print("Invalid indices. Exiting.")
        return

    lat = float(input("Enter latitude: ").strip())
    lon = float(input("Enter longitude: ").strip())
    
    frame_files = []
    for i in range(start_idx, end_idx + 1):
        vt, fname, fcst = valid_list[i]
        print(f"Processing valid time: {vt.strftime('%Y-%m-%d %HZ')} from file {fname}")
        file_path = download_model_file(fname, download_base)
        if not file_path:
            print("File download failed; skipping this frame.")
            continue
        frame_file = f"frame_{i:03d}.png"
        plot_skewt(file_path, lat, lon, model, output_file=frame_file)
        frame_files.append(frame_file)
    
    frames = []
    for frame_file in frame_files:
        frames.append(imageio.imread(frame_file))
    gif_filename = f"{model}_skewt_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.gif"
    imageio.mimsave(gif_filename, frames, duration=5, loop=0)
    print(f"GIF saved as {gif_filename}")

if __name__ == "__main__":
    main()
