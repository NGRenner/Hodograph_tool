# Use this script if you want load your own .grib2 files to make a hodograph and a vertical wind barb profile. 
# itll ask for a for 


import os
import requests
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.interpolate import griddata, interp1d
import pandas as pd
import matplotlib.colors as mcolors
from metpy.plots import Hodograph
from metpy.calc import wind_speed, wind_direction, bunkers_storm_motion, storm_relative_helicity
from metpy.units import units
import pygrib


def get_elevation_pygrib(file_path, lat_target, lon_target):
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

def calculate_critical_angle(u_profile, v_profile, z_agl_km, storm_motion):
    """
    Calculate the critical angle between the storm-relative wind at the surface
    and the 0–0.5 km shear vector. Matches SHARPpy's definition.

    Parameters:
        u_profile (pint.Quantity): u-component wind profile [m/s]
        v_profile (pint.Quantity): v-component wind profile [m/s]
        z_agl_km (ndarray): heights above ground level [km]
        storm_motion (tuple of pint.Quantity): storm motion (u, v) in [m/s]

    Returns:
        float: critical angle in degrees
    """
    # Trim all to same length to prevent mismatch
    min_len = min(len(u_profile), len(v_profile), len(z_agl_km))
    u_vals = u_profile[:min_len].to('m/s').magnitude
    v_vals = v_profile[:min_len].to('m/s').magnitude
    z_vals = np.asarray(z_agl_km[:min_len])

    # Interpolation functions for u/v
    interp_u = interp1d(z_vals, u_vals, bounds_error=False, fill_value='extrapolate')
    interp_v = interp1d(z_vals, v_vals, bounds_error=False, fill_value='extrapolate')

    # Wind at surface and 0.5 km
    u0 = interp_u(0.0)
    v0 = interp_v(0.0)
    u500 = interp_u(0.5)
    v500 = interp_v(0.5)

    # Shear vector (0–0.5 km)
    shear_u = u500 - u0
    shear_v = v500 - v0
    shear_mag = np.hypot(shear_u, shear_v)

    # Storm-relative wind at surface
    storm_u = storm_motion[0].to('m/s').magnitude
    storm_v = storm_motion[1].to('m/s').magnitude
    srw_u = u0 - storm_u
    srw_v = v0 - storm_v
    srw_mag = np.hypot(srw_u, srw_v)

    # Critical angle = angle between shear vector and SRW vector
    dot = shear_u * srw_u + shear_v * srw_v
    cos_theta = dot / (shear_mag * srw_mag)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
   
    angle_deg = np.degrees(angle_rad)

    # SHARPpy-style: restrict to smallest angle between vectors
    if angle_deg > 90:
        angle_deg = 180 - angle_deg

    return angle_deg


def plot_hodograph(data_file, lat, lon, model, output_file=None):
     # === Load GRIB data ===
    ds_u = xr.open_dataset(data_file, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}}, indexpath=None)
    ds_v = xr.open_dataset(data_file, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}}, indexpath=None)
    ds_z = xr.open_dataset(data_file, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh'}}, indexpath=None)

    u, v, z = ds_u['u'], ds_v['v'], ds_z['gh']
    pressure = ds_u['isobaricInhPa']
    lat_isobaric = ds_u['latitude'].values
    lon_isobaric = ds_u['longitude'].values

    # === Surface Elevation (fake fallback) ===
    station_elev_m = get_elevation_pygrib(data_file, lat, lon) or 0.0

    # === Time ===
    rt = pd.to_datetime(ds_u.time.values)
    vt = pd.to_datetime(ds_u.valid_time.values)
    if lon_isobaric.min() >= 0 and lon_isobaric.max() > 180 and lon < 0:
        lon += 360

    # === Pressure levels to extract ===
    levels = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
              750, 725, 700, 675, 650, 625, 600, 575, 550, 525,
              500, 475, 450, 425, 400, 375, 350, 325, 300, 275,
              250, 225, 200, 175, 150, 125, 100]

    # === Grid setup ===
    if lat_isobaric.ndim == 1 and lon_isobaric.ndim == 1:
        lat_grid, lon_grid = np.meshgrid(lat_isobaric, lon_isobaric, indexing='ij')
    else:
        lat_grid, lon_grid = lat_isobaric, lon_isobaric

    # === Interpolation ===
    u_interp, v_interp, z_interp, levels_used = [], [], [], []
    for lev in tqdm(levels, desc="Interpolating Wind/Height Data"):
        if lev not in pressure.values: continue
        idx = np.abs(pressure.values - lev).argmin()
        u_level, v_level, z_level = u.isel(isobaricInhPa=idx).values, v.isel(isobaricInhPa=idx).values, z.isel(isobaricInhPa=idx).values
        u_pt = griddata((lat_grid.ravel(), lon_grid.ravel()), u_level.ravel(), (lat, lon), method='nearest')
        v_pt = griddata((lat_grid.ravel(), lon_grid.ravel()), v_level.ravel(), (lat, lon), method='nearest')
        z_pt = griddata((lat_grid.ravel(), lon_grid.ravel()), z_level.ravel(), (lat, lon), method='nearest')
        u_interp.append(u_pt)
        v_interp.append(v_pt)
        z_interp.append(z_pt)
        levels_used.append(lev)

    # === Clean below-ground levels ===
    u_interp = np.array(u_interp)
    v_interp = np.array(v_interp)
    z_interp = np.array(z_interp)
    levels_used = np.array(levels_used)

    z_m = (z_interp / 9.80665) * 10
    z_agl_m = z_m - station_elev_m
    valid = z_agl_m >= 0
    u_interp = u_interp[valid]
    v_interp = v_interp[valid]
    z_interp = z_interp[valid]
    levels_used = levels_used[valid]

    z_m = (z_interp / 9.80665) * 10
    z_agl_m = z_m - station_elev_m
    z_agl_km = z_agl_m / 1000.0

    sort_idx = np.argsort(z_agl_m)
    u_interp = u_interp[sort_idx]
    v_interp = v_interp[sort_idx]
    z_agl_m = z_agl_m[sort_idx]
    z_agl_km = z_agl_km[sort_idx]
    pressure_profile = levels_used[sort_idx] * units.hPa


  

    # === Units ===
    u_profile = u_interp * units('m/s')
    v_profile = v_interp * units('m/s')
    height_profile = z_agl_m * units.meter

    u_knots = u_profile.to('knots').magnitude
    v_knots = v_profile.to('knots').magnitude

    interp_func_u = interp1d(z_agl_km, u_interp, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_func_v = interp1d(z_agl_km, v_interp, kind='linear', bounds_error=False, fill_value="extrapolate")

    # === Interpolated surface wind (for plotting) ===
    u_0 = float((interp_func_u(0.0) * units('m/s')).to('knots').magnitude)
    v_0 = float((interp_func_v(0.0) * units('m/s')).to('knots').magnitude)
    u_knots = np.insert(u_knots, 0, u_0)
    v_knots = np.insert(v_knots, 0, v_0)
    z_agl_km = np.insert(z_agl_km, 0, 0.0)

    # === Storm motion ===
    rm, lm, mean = bunkers_storm_motion(pressure_profile, u_profile, v_profile, height_profile)
    crit_angle = calculate_critical_angle(u_profile, v_profile, z_agl_km, rm)

    # === Plot setup ===
    fig, (ax_hodo, ax_barbs) = plt.subplots(1, 2, figsize=(9, 6), width_ratios=[3, 0.3])
    fig.subplots_adjust(wspace=0.15)
    h = Hodograph(ax_hodo, component_range=80)
    h.add_grid(increment=10)
    ax_hodo.set_xlim(-80, 80)
    ax_hodo.set_ylim(-80, 80)

    # === SHARPpy-style ring labels ===
    for ws in range(10, 90, 10):
        ax_hodo.text(ws, 0, f"{ws}", ha='center', va='bottom', fontsize=9, color='gray')
        ax_hodo.text(-ws, 0, f"{ws}", ha='center', va='bottom', fontsize=9, color='gray')
        ax_hodo.text(0, ws, f"{ws}", ha='left', va='center', fontsize=9, color='gray')
        ax_hodo.text(0, -ws, f"{ws}", ha='left', va='center', fontsize=9, color='gray')

    # === Plot SRH polygon ===
    srh_val, _, _ = storm_relative_helicity(height_profile, u_profile, v_profile, 3000 * units.meter, storm_u=rm[0], storm_v=rm[1])
    idx_3km = np.argmin(np.abs(z_agl_km - 3))
    u_srh_plot = u_knots[:idx_3km + 1]
    v_srh_plot = v_knots[:idx_3km + 1]
    rm_u = rm[0].to('knots').magnitude
    rm_v = rm[1].to('knots').magnitude
    poly_u = np.concatenate([[rm_u], u_srh_plot, [rm_u]])
    poly_v = np.concatenate([[rm_v], v_srh_plot, [rm_v]])
    ax_hodo.fill(poly_u, poly_v, color='lightblue', alpha=0.4, zorder=2)

    # === SRW Line ===
    u_target = u_profile[idx_3km].to('knots').magnitude
    v_target = v_profile[idx_3km].to('knots').magnitude
    ax_hodo.plot([u_0, rm_u], [v_0, rm_v], color='blue')
    ax_hodo.plot([rm_u, u_target], [rm_v, v_target], color='blue')

    # === Storm markers ===
    lm_u, lm_v = lm[0].to('knots').magnitude, lm[1].to('knots').magnitude
    mean_u, mean_v = mean[0].to('knots').magnitude, mean[1].to('knots').magnitude
    ax_hodo.scatter(rm_u, rm_v, edgecolors='red', facecolors='none', s=60, zorder=10)
    ax_hodo.scatter(lm_u, lm_v, edgecolors='blue', facecolors='none', s=60, zorder=10)
    ax_hodo.scatter(mean_u, mean_v, edgecolors='green', facecolors='none', s=60, zorder=10)

    # === Plot profile line ===
    cmap = plt.get_cmap("Spectral")
    norm = mcolors.Normalize(vmin=0, vmax=len(u_knots) - 1)
    for i in range(len(u_knots) - 1):
        h.plot(u_knots[i:i+2], v_knots[i:i+2], color=cmap(norm(i)), linewidth=2)

    # === Barbs ===
    idx_max = np.argmax(z_agl_km >= 12) if np.any(z_agl_km >= 12) else len(z_agl_km)
    u_barbs = u_profile[:idx_max].to('knots').magnitude
    v_barbs = v_profile[:idx_max].to('knots').magnitude
    z_barbs = z_agl_km[:idx_max]
    ax_barbs.barbs(np.zeros_like(z_barbs), z_barbs, u_barbs, v_barbs, length=6)
    ax_barbs.set_ylim(ax_barbs.get_ylim()[0] - 0.5, 12)
    ax_barbs.set_xlim(-1, 1)
    ax_barbs.set_xticks([])
    ax_barbs.set_ylabel("Height AGL (km)")

    # === Altitude markers ===
    for alt in range(1, 9):
        if np.min(z_agl_km) <= alt <= np.max(z_agl_km):
            ua = (interp_func_u(alt) * units('m/s')).to('knots').magnitude
            va = (interp_func_v(alt) * units('m/s')).to('knots').magnitude
            ax_hodo.scatter(ua, va, color='black', s=30)
            ax_hodo.text(ua + 1.0, va, f"{alt} km", fontsize=9, color='black', ha='left', va='center')

    # === Annotations ===
    ax_hodo.text(0.02, 0.02, f"0–3 km SRH: {srh_val:.0f} m²/s²", transform=ax_hodo.transAxes,
                 fontsize=10, color='navy', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax_hodo.text(0.02, 0.08, f"Critical Angle: {crit_angle:.1f}°", transform=ax_hodo.transAxes,
                 fontsize=9, color='darkgreen', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    ax_hodo.text(rm_u + 1, rm_v, 'BRM', color='red', fontsize=9, va='center')
    ax_hodo.text(lm_u + 1, lm_v, 'BLM', color='blue', fontsize=9, va='center')
    ax_hodo.text(mean_u + 1, mean_v, '0-6km MW', color='green', fontsize=9, va='center')

    fcst_hour = int((vt - rt).total_seconds() // 3600)
    metadata_text = (
        f"{model.upper()} {rt.strftime('%Y-%m-%d %HZ')}, F{fcst_hour:03d}\n"
        f"VALID: {vt.strftime('%a %Y-%m-%d %HZ')}\n"
        f"AT: {lat:.2f}°N, {lon:.2f}°{'W' if lon < 0 else 'E'}")
    ax_hodo.text(0.02, 0.98, metadata_text, transform=ax_hodo.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', boxstyle="round,pad=0.3"), color='white')

    if output_file:
        plt.savefig(output_file, dpi=150)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    Tk().withdraw()  # Hide root window
    file_path = askopenfilename(title="Select GRIB2 File", filetypes=[("GRIB2 files", "*.grb2"), ("All files", "*.*")])
    
    if not file_path:
        print("No file selected. Exiting.")
        exit()

    lat = float(input("Enter latitude: ").strip())
    lon = float(input("Enter longitude: ").strip())

    # Ask for model manually
    model = input("Enter model name (e.g., rap, nam, gfs): ").strip().lower()
    plot_hodograph(file_path, lat, lon, model)