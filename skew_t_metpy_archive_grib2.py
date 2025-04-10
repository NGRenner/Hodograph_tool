# THIS IS OUTDATED ONLY CAN USE SINGLE DATA FILES

#!/usr/bin/env python3
import os
import requests
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.interpolate import griddata
import pandas as pd
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



def get_elevation_pygrib(file_path, lat_target, lon_target):
    """
    Grab surface elevation (or geopotential height) from the GRIB2 file
    using pygrib. This is used to compute AGL.
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
        print("Warning: No orography/hgt field found in the file.")
        elev_input = input("Please enter the surface elevation (in meters): ")
        try:
            elev = float(elev_input)
        except Exception:
            print("Invalid input; defaulting to 0 m.")
            elev = 0.0
        return elev
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
    Read temperature, RH, wind, and geopotential height from GRIB2 file,
    interpolate to a single lat/lon, and plot a Skew-T with CAPE/CIN shading,
    parcel profile, and wind barbs.
    """
    # Open relevant fields using cfgrib
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

    # Extract data arrays
    t = ds_t["t"]      # K
    rh = ds_rh["r"]    # %
    u = ds_u["u"]      # m/s
    v = ds_v["v"]      # m/s
    z = ds_z["gh"]     # geopotential
    pressure = ds_t["isobaricInhPa"]
    lat_isobaric = ds_t["latitude"].values
    lon_isobaric = ds_t["longitude"].values

    # Determine surface elevation for AGL
    station_elev_m = get_elevation_pygrib(data_file, lat, lon)
    if station_elev_m is None:
        print("Surface terrain elevation not found; defaulting to 0 m.")
        station_elev_m = 0.0
    print(f"Surface elevation at {lat:.2f}°N, {lon:.2f}° is ~{station_elev_m:.1f} m")

    rt = pd.to_datetime(ds_t.time.values)
    vt = pd.to_datetime(ds_t.valid_time.values)

    if lon_isobaric.min() >= 0 and lon_isobaric.max() > 180 and lon < 0:
        lon = lon + 360

    # Define desired pressure levels
    levels = [
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
        750, 725, 700, 675, 650, 625, 600, 575, 550, 525,
        500, 475, 450, 425, 400, 375, 350, 325, 300, 275,
        250, 225, 200, 175, 150, 125, 100
    ]

    # Create coordinate grids for interpolation
    if lat_isobaric.ndim == 1 and lon_isobaric.ndim == 1:
        lat_grid, lon_grid = np.meshgrid(lat_isobaric, lon_isobaric, indexing='ij')
    else:
        lat_grid, lon_grid = lat_isobaric, lon_isobaric

    # Lists to store interpolated data
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

    T_kelvin = t_interp * units.K
    RH_frac = (rh_interp / 100.0)
    Td_kelvin = dewpoint_from_relative_humidity(T_kelvin, RH_frac)

    u_profile = (u_interp * units.meter / units.second)
    v_profile = (v_interp * units.meter / units.second)

    z_agl_m = z_m - station_elev_m
    sort_idx = np.argsort(p_profile.m, kind='mergesort')[::-1]
    p_sorted = p_profile[sort_idx]
    T_sorted = T_kelvin[sort_idx]
    Td_sorted = Td_kelvin[sort_idx]
    u_sorted = u_profile[sort_idx]
    v_sorted = v_profile[sort_idx]
    z_agl_sorted = z_agl_m[sort_idx] * units.meter

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
    lcl_height_m = np.interp(lcl_p.m, p_sorted.m[::-1], z_agl_sorted.m[::-1])
    print(f"LCL: {lcl_p:.1f}, {lcl_t.to('degC'):.1f} => ~{lcl_height_m:.0f} m AGL")

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
                f"LCL: {lcl_height_m:.0f} m AGL")
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
    skew.ax.text(0.98, 1.01, metadata_text, transform=skew.ax.transAxes, ha='right', va='bottom',
                 bbox=dict(facecolor='black', alpha=0.8, edgecolor='none'),
                 color='white', fontsize=9)

    plt.legend(loc='best', fontsize=9)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    Tk().withdraw()  # Hide root window
    file_path = askopenfilename(title="Select GRIB2 File", filetypes=[("GRIB2 files", "*.grib2"), ("All files", "*.*")])
    
    if not file_path:
        print("No file selected. Exiting.")
        exit()

    lat = float(input("Enter latitude: ").strip())
    lon = float(input("Enter longitude: ").strip())

    # Ask for model manually
    model = input("Enter model name (e.g., rap, nam, gfs): ").strip().lower()
    plot_skewt(file_path, lat, lon, model)