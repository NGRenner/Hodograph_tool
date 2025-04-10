#!/usr/bin/env python3
"""
Combined MetPy plotting script for GRIB2 files.
This version opens individual datasets for 'u', 'v', 'gh', 't', and 'r',
merges them, and loads the dataset into memory with Dask. For each selected
time slice, it produces a combined figure with the Skew-T/Log-P diagram on the
left and the hodograph (with all your original labels and markers) on the right.
"""

import os
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.interpolate import griddata, interp1d
import pandas as pd
import matplotlib.colors as mcolors
from metpy.plots import Hodograph, SkewT
from metpy.calc import wind_speed, wind_direction, bunkers_storm_motion, storm_relative_helicity
from metpy.calc import dewpoint_from_relative_humidity, parcel_profile, lcl, cape_cin
from metpy.units import units
import pygrib
from dask.diagnostics import ProgressBar
import matplotlib.gridspec as gridspec

# ---------------------
# Utility Functions
# ---------------------
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
    # Trim arrays to the same length
    min_len = min(len(u_profile), len(v_profile), len(z_agl_km))
    u_vals = u_profile[:min_len].to('m/s').magnitude
    v_vals = v_profile[:min_len].to('m/s').magnitude
    z_vals = np.asarray(z_agl_km[:min_len])
    interp_u = interp1d(z_vals, u_vals, bounds_error=False, fill_value='extrapolate')
    interp_v = interp1d(z_vals, v_vals, bounds_error=False, fill_value='extrapolate')
    u0 = interp_u(0.0)
    v0 = interp_v(0.0)
    u500 = interp_u(0.5)
    v500 = interp_v(0.5)
    shear_u = u500 - u0
    shear_v = v500 - v0
    shear_mag = np.hypot(shear_u, shear_v)
    storm_u = storm_motion[0].to('m/s').magnitude
    storm_v = storm_motion[1].to('m/s').magnitude
    srw_u = u0 - storm_u
    srw_v = v0 - storm_v
    srw_mag = np.hypot(srw_u, srw_v)
    dot = shear_u * srw_u + shear_v * srw_v
    cos_theta = dot / (shear_mag * srw_mag)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    return angle_deg

# ---------------------
# Combined Plotting Function
# ---------------------
def plot_combined(ds, file_path, lat, lon, model, output_file=None):
    """
    Creates a combined figure with three panels:
      - Left panel (spanning both rows): Skew-T/Log-P diagram.
      - Top-right panel: Hodograph (with markers, BRM/BLM, height labels, etc.).
      - Bottom-right panel: A text area for calculated variables.
    Data is taken from the pre-sliced dataset ds.
    """
    # ===== Common Data and Metadata =====
    station_elev_m = get_elevation_pygrib(file_path, lat, lon) or 0.0
    rt = pd.to_datetime(np.atleast_1d(ds.time.values)[0])
    vt = pd.to_datetime(np.atleast_1d(ds.valid_time.values)[0])
    lat_isobaric = ds['latitude'].values
    lon_isobaric = ds['longitude'].values
    if lon_isobaric.min() >= 0 and lon_isobaric.max() > 180 and lon < 0:
        lon += 360

    # ===== Figure and GridSpec Setup =====
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[2, 1], height_ratios=[2, 1],
                           wspace=0.3, hspace=0.3)
    # Left panel: Skew-T (spanning both rows of column 0)
    ax_skew = fig.add_subplot(gs[:, 0])
    # Top-right: Hodograph
    ax_hodo = fig.add_subplot(gs[0, 1])
    # Bottom-right: Calculated variables (text); disable axis for clarity
    ax_calc = fig.add_subplot(gs[1, 1])
    ax_calc.axis('off')

    # ===== Skew-T/Log-P Plotting (Left Panel) =====
    # Create a SkewT object on ax_skew.
    skew = SkewT(fig, subplot=(2,2,1), rotation=45)
    # Note: Using subplot=(2,2,1) will assign the left-top cell. To make it span both rows,
    # we can manually adjust its position to fill the entire left column.
    # For simplicity, we then set the position of ax_skew equal to the SkewT axis.
    # (Alternatively, if your MetPy version supports the "ax" keyword, try: skew = SkewT(ax=ax_skew, rotation=45))
    # Here we assume the SkewT object was created successfully.
    
    # Interpolate for Skew-T variables.
    levels = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
              750, 725, 700, 675, 650, 625, 600, 575, 550, 525,
              500, 475, 450, 425, 400, 375, 350, 325, 300, 275,
              250, 225, 200, 175, 150, 125, 100]
    if lat_isobaric.ndim == 1 and lon_isobaric.ndim == 1:
        lat_grid, lon_grid = np.meshgrid(lat_isobaric, lon_isobaric, indexing='ij')
    else:
        lat_grid, lon_grid = lat_isobaric, lon_isobaric

    t = ds['t']    # Temperature (K)
    r = ds['r']    # RH (%)
    t_interp, r_interp, u_interp_skew, v_interp_skew, z_interp_skew, levs_skew = [], [], [], [], [], []
    for lev in tqdm(levels, desc="Interpolating SkewT Data"):
        if lev not in ds['isobaricInhPa'].values:
            continue
        idx = np.abs(ds['isobaricInhPa'].values - lev).argmin()
        t_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        t.isel(isobaricInhPa=idx).values.ravel(), (lat, lon), method='nearest')
        r_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        r.isel(isobaricInhPa=idx).values.ravel(), (lat, lon), method='nearest')
        u_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        ds['u'].isel(isobaricInhPa=idx).values.ravel(), (lat, lon), method='nearest')
        v_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        ds['v'].isel(isobaricInhPa=idx).values.ravel(), (lat, lon), method='nearest')
        z_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        ds['gh'].isel(isobaricInhPa=idx).values.ravel(), (lat, lon), method='nearest')
        t_interp.append(t_pt)
        r_interp.append(r_pt)
        u_interp_skew.append(u_pt)
        v_interp_skew.append(v_pt)
        z_interp_skew.append(z_pt)
        levs_skew.append(lev)
    t_interp = np.array(t_interp)
    r_interp = np.array(r_interp)
    u_interp_skew = np.array(u_interp_skew)
    v_interp_skew = np.array(v_interp_skew)
    z_interp_skew = np.array(z_interp_skew)
    levs_skew = np.array(levs_skew)

    p_profile = levs_skew * units.hPa
    z_m_skew = (z_interp_skew / 9.80665) * 10.0
    T_kelvin = t_interp * units.K
    RH_frac = (r_interp / 100.0)
    Td_kelvin = dewpoint_from_relative_humidity(T_kelvin, RH_frac)
    u_profile_skew = u_interp_skew * units('m/s')
    v_profile_skew = v_interp_skew * units('m/s')
    z_agl_m_skew = z_m_skew - station_elev_m
    sort_idx = np.argsort(p_profile.m, kind='mergesort')[::-1]
    p_sorted_skew = p_profile[sort_idx]
    T_sorted = T_kelvin[sort_idx]
    Td_sorted = Td_kelvin[sort_idx]
    u_sorted_skew = u_profile_skew[sort_idx]
    v_sorted_skew = v_profile_skew[sort_idx]
    z_agl_sorted_skew = (z_agl_m_skew[sort_idx]) * units.meter

    # Plot Skew-T
    skew.plot(p_sorted_skew, T_sorted.to('degC'), color='red', linewidth=2, label='Temperature')
    skew.plot(p_sorted_skew, Td_sorted.to('degC'), color='green', linewidth=2, label='Dewpoint')
    step = max(1, len(p_sorted_skew) // 15)
    skew.plot_barbs(p_sorted_skew[::step], u_sorted_skew[::step], v_sorted_skew[::step])
    p_sfc = p_sorted_skew[0]
    T_sfc = T_sorted[0]
    Td_sfc = Td_sorted[0]
    lcl_p, lcl_t = lcl(p_sfc, T_sfc, Td_sfc)
    lcl_height_m = np.interp(lcl_p.m, p_sorted_skew.m[::-1], z_agl_sorted_skew.m[::-1])
    prof = parcel_profile(p_sorted_skew, T_sfc, Td_sfc)
    skew.plot(p_sorted_skew, prof.to('degC'), color='black', linestyle='--', linewidth=2, label='Parcel')
    cape, cin = cape_cin(p_sorted_skew, T_sorted, Td_sorted, prof)
    skew.shade_cin(p_sorted_skew, T_sorted.to('degC'), prof.to('degC'), color='blue', alpha=0.2)
    skew.shade_cape(p_sorted_skew, T_sorted.to('degC'), prof.to('degC'), color='red', alpha=0.2)
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    skew.ax.set_title("Skew-T/Log-P Diagram", loc='left', fontsize=11)

    # ===== Hodograph Plotting (Top-Right Panel) =====
    u = ds['u']
    v = ds['v']
    z = ds['gh']
    pressure = ds['isobaricInhPa']
    u_interp, v_interp, z_interp, levels_used = [], [], [], []
    for lev in tqdm(levels, desc="Interpolating Hodograph Data"):
        if lev not in pressure.values:
            continue
        idx = np.abs(pressure.values - lev).argmin()
        u_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        u.isel(isobaricInhPa=idx).values.ravel(), (lat, lon), method='nearest')
        v_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        v.isel(isobaricInhPa=idx).values.ravel(), (lat, lon), method='nearest')
        z_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        z.isel(isobaricInhPa=idx).values.ravel(), (lat, lon), method='nearest')
        u_interp.append(u_pt)
        v_interp.append(v_pt)
        z_interp.append(z_pt)
        levels_used.append(lev)
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
    u_profile = u_interp * units('m/s')
    v_profile = v_interp * units('m/s')
    height_profile = z_agl_m * units.meter
    u_knots = u_profile.to('knots').magnitude
    v_knots = v_profile.to('knots').magnitude
    interp_func_u = interp1d(z_agl_km, u_interp, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_func_v = interp1d(z_agl_km, v_interp, kind='linear', bounds_error=False, fill_value="extrapolate")
    u_0 = float((interp_func_u(0.0) * units('m/s')).to('knots').magnitude)
    v_0 = float((interp_func_v(0.0) * units('m/s')).to('knots').magnitude)
    u_knots = np.insert(u_knots, 0, u_0)
    v_knots = np.insert(v_knots, 0, v_0)
    z_agl_km = np.insert(z_agl_km, 0, 0.0)
    rm, lm, mean = bunkers_storm_motion(pressure_profile, u_profile, v_profile, height_profile)
    crit_angle = calculate_critical_angle(u_profile, v_profile, z_agl_km, rm)
    hodo = Hodograph(ax_hodo, component_range=80)
    hodo.add_grid(increment=10)
    ax_hodo.set_xlim(-80, 80)
    ax_hodo.set_ylim(-80, 80)
    for ws in range(10, 90, 10):
        ax_hodo.text(ws, 0, f"{ws}", ha='center', va='bottom', fontsize=9, color='gray')
        ax_hodo.text(-ws, 0, f"{ws}", ha='center', va='bottom', fontsize=9, color='gray')
        ax_hodo.text(0, ws, f"{ws}", ha='left', va='center', fontsize=9, color='gray')
        ax_hodo.text(0, -ws, f"{ws}", ha='left', va='center', fontsize=9, color='gray')
    srh_val, _, _ = storm_relative_helicity(height_profile, u_profile, v_profile, 3000 * units.meter,
                                              storm_u=rm[0], storm_v=rm[1])
    idx_3km = np.argmin(np.abs(z_agl_km - 3))
    u_srh_plot = u_knots[:idx_3km + 1]
    v_srh_plot = v_knots[:idx_3km + 1]
    rm_u = rm[0].to('knots').magnitude
    rm_v = rm[1].to('knots').magnitude
    poly_u = np.concatenate([[rm_u], u_srh_plot, [rm_u]])
    poly_v = np.concatenate([[rm_v], v_srh_plot, [rm_v]])
    ax_hodo.fill(poly_u, poly_v, color='lightblue', alpha=0.4, zorder=2)
    u_target = u_knots[idx_3km]
    v_target = v_knots[idx_3km]
    ax_hodo.plot([u_0, rm_u], [v_0, rm_v], color='blue')
    ax_hodo.plot([rm_u, u_target], [rm_v, v_target], color='blue')
    lm_u, lm_v = lm[0].to('knots').magnitude, lm[1].to('knots').magnitude
    mean_u, mean_v = mean[0].to('knots').magnitude, mean[1].to('knots').magnitude
    ax_hodo.scatter(rm_u, rm_v, edgecolors='red', facecolors='none', s=60, zorder=10)
    ax_hodo.scatter(lm_u, lm_v, edgecolors='blue', facecolors='none', s=60, zorder=10)
    ax_hodo.scatter(mean_u, mean_v, edgecolors='green', facecolors='none', s=60, zorder=10)
    cmap = plt.get_cmap("Spectral")
    norm = mcolors.Normalize(vmin=0, vmax=len(u_knots) - 1)
    for i in range(len(u_knots) - 1):
        hodo.plot(u_knots[i:i+2], v_knots[i:i+2], color=cmap(norm(i)), linewidth=2)
    for alt in range(1, 9):
        if np.min(z_agl_km) <= alt <= np.max(z_agl_km):
            ua = (interp_func_u(alt) * units('m/s')).to('knots').magnitude
            va = (interp_func_v(alt) * units('m/s')).to('knots').magnitude
            ax_hodo.scatter(ua, va, color='black', s=30, zorder=12)
            ax_hodo.text(ua + 1.0, va, f"{alt} km", fontsize=9, color='black', ha='left', va='center', zorder=12)
    ax_hodo.text(0.02, 0.02, f"0–3 km SRH: {srh_val:.0f} m²/s²", transform=ax_hodo.transAxes,
                 fontsize=10, color='navy', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), zorder=12)
    ax_hodo.text(0.02, 0.08, f"Critical Angle: {crit_angle:.1f}°", transform=ax_hodo.transAxes,
                 fontsize=9, color='darkgreen', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'), zorder=12)
    ax_hodo.text(rm_u + 1, rm_v, 'BRM', color='red', fontsize=9, va='center', zorder=12)
    ax_hodo.text(lm_u + 1, lm_v, 'BLM', color='blue', fontsize=9, va='center', zorder=12)
    ax_hodo.text(mean_u + 1, mean_v, '0-6km MW', color='green', fontsize=9, va='center', zorder=12)
    ax_hodo.text(0.02, 0.98,
                 f"{model.upper()} {rt.strftime('%Y-%m-%d %HZ')}, F{int((vt-rt).total_seconds()//3600):03d}\n"
                 f"VALID: {vt.strftime('%a %Y-%m-%d %HZ')}\nAT: {lat:.2f}°N, {lon:.2f}°{'W' if lon<0 else 'E'}",
                 transform=ax_hodo.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', boxstyle="round,pad=0.3"), color='white', zorder=12)

    # ===== Calculated Variables Panel (Bottom-Right) =====
    calc_text = (
        f"CAPE: {cape.to('joules/kilogram').magnitude:.0f} J/kg\n"
        f"CIN: {cin.to('joules/kilogram').magnitude:.0f} J/kg\n"
        f"LCL: {lcl_height_m:.0f} m AGL\n"
        f"0-3 km SRH: {srh_val:.0f} m²/s²\n"
        f"Crit. Angle: {crit_angle:.1f}°"
    )
    ax_calc.text(0.05, 0.95, calc_text, ha='left', va='top', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    fig.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# ---------------------
# Main Block
# ---------------------
if __name__ == "__main__":
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename

    root = Tk()
    root.withdraw()
    root.update()
    root.lift()
    root.attributes('-topmost', True)
    mode = input("Process (1) a single timestamp or (2) multiple timestamps? Enter 1 or 2: ").strip()
    file_path = askopenfilename(title="Select GRIB2 File", filetypes=[("GRIB2 files", "*.grb2"), ("All files", "*.*")])
    if not file_path:
        print("No file selected. Exiting.")
        exit()
    lat = float(input("Enter latitude: ").strip())
    lon = float(input("Enter longitude: ").strip())
    model = input("Enter model name (e.g., rap, nam, gfs): ").strip().lower()
    print("Opening dataset. This may take a while for large files.")
    ds_u = xr.open_dataset(
        file_path, 
        engine='cfgrib',
        chunks={'time': 1, 'isobaricInhPa': 10},
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}},
        indexpath=None
    ).drop_vars("step", errors="ignore")
    ds_v = xr.open_dataset(
        file_path, 
        engine='cfgrib',
        chunks={'time': 1, 'isobaricInhPa': 10},
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}},
        indexpath=None
    ).drop_vars("step", errors="ignore")
    ds_gh = xr.open_dataset(
        file_path, 
        engine='cfgrib',
        chunks={'time': 1, 'isobaricInhPa': 10},
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh'}},
        indexpath=None
    ).drop_vars("step", errors="ignore")
    ds_t = xr.open_dataset(
        file_path, engine='cfgrib',
        chunks={'time': 1, 'isobaricInhPa': 10},
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 't'}},
        indexpath=None, decode_timedelta=False
    ).drop_vars("step", errors="ignore")
    ds_r = xr.open_dataset(
        file_path, engine='cfgrib',
        chunks={'time': 1, 'isobaricInhPa': 10},
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'r'}},
        indexpath=None, decode_timedelta=False
    ).drop_vars("step", errors="ignore")
    ds = xr.merge([ds_u, ds_v, ds_gh, ds_t, ds_r])
    if "time" not in ds.dims:
        ds = ds.expand_dims("time")
    with ProgressBar():
        ds = ds.load()
    times = np.atleast_1d(pd.to_datetime(ds.time.values))
    print("Processing timestamps:", times)
    if mode == "2":
        if len(times) == 1:
            print("File contains only a single timestamp; processing as single timestamp.")
            mode = "1"
        else:
            print("Available time stamps:")
            for i, t in enumerate(times):
                print(f"{i}: {t}")
            selection = input("Enter the indices of the time stamps to process (comma-separated, e.g., 0,2,3): ").strip()
            try:
                selected_indices = [int(s.strip()) for s in selection.split(',')]
            except Exception as e:
                print("Invalid input for time indices, exiting.")
                exit()
            for idx in selected_indices:
                selected_time = pd.to_datetime(times[idx])
                time_str = selected_time.strftime('%Y%m%d_%H%M')
                out_filename = f"combined_{time_str}.png"
                print(f"Processing timestamp {selected_time} (index {idx}); saving plot to {out_filename}")
                ds_time = ds.isel(time=idx)
                plot_combined(ds_time, file_path, lat, lon, model, output_file=out_filename)
            print("All selected timestamps processed.")
    if mode == "1":
        print(f"Processing single timestamp: {times[0]}")
        ds_time = ds.isel(time=0) if "time" in ds.dims else ds
        plot_combined(ds_time, file_path, lat, lon, model)
