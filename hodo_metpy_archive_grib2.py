# updated as of 3.21.2025

import os
import requests
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.colors as mcolors
from metpy.plots import Hodograph
from metpy.calc import wind_speed, wind_direction, bunkers_storm_motion, storm_relative_helicity
from metpy.units import units
import tkinter as tk
from tkinter import filedialog
import pygrib
import cfgrib


def get_elevation_pygrib(file_path, lat_target, lon_target):
    """
    Opens the GRIB2 file using pygrib, selects the orography field,
    and returns the elevation (in meters) at the nearest grid point to
    the specified latitude and longitude.
    """
    grbs = pygrib.open(file_path)
    # Select the orography field; typically shortName is 'orog' and typeOfLevel is 'surface'
    orog_msgs = grbs.select(shortName='orog', typeOfLevel='surface')
    if len(orog_msgs) == 0:
        raise ValueError("No orography field found in the file.")
    grb = orog_msgs[0]
    print("Using GRIB message for orography:")
    print(grb)
    
    # Get the latitude, longitude, and elevation arrays
    lats, lons = grb.latlons()
    elevations = grb.values

    # Adjust if the file uses 0-360 longitudes
    if lons.min() >= 0 and lons.max() > 180 and lon_target < 0:
        lon_target = lon_target + 360

    # Nearest-neighbor: compute distance to all grid points and get the minimum index
    dist = np.sqrt((lats - lat_target)**2 + (lons - lon_target)**2)
    min_index = np.unravel_index(np.argmin(dist), dist.shape)
    elevation_at_point = elevations[min_index]
    return elevation_at_point
def calculate_critical_angle(u_profile, v_profile, z_agl_km, storm_motion):
    # Get surface wind
    idx_sfc = np.argmin(np.abs(z_agl_km - 0))
    u_sfc = u_profile[idx_sfc].to('m/s').magnitude
    v_sfc = v_profile[idx_sfc].to('m/s').magnitude

    # Get 1 km wind
    idx_1km = np.argmin(np.abs(z_agl_km - 1))
    u_1km = u_profile[idx_1km].to('m/s').magnitude
    v_1km = v_profile[idx_1km].to('m/s').magnitude

    # Calculate shear vector (0–1 km)
    shear_u = u_1km - u_sfc
    shear_v = v_1km - v_sfc

    # Storm motion relative to sfc wind
    rm_u = storm_motion[0].to('m/s').magnitude
    rm_v = storm_motion[1].to('m/s').magnitude

    storm_rel_u = rm_u - u_sfc
    storm_rel_v = rm_v - v_sfc

    # Compute angle between vectors
    shear_vec = np.array([shear_u, shear_v])
    storm_vec = np.array([storm_rel_u, storm_rel_v])

    shear_mag = np.linalg.norm(shear_vec)
    storm_mag = np.linalg.norm(storm_vec)

    if shear_mag == 0 or storm_mag == 0:
        return np.nan

    cos_theta = np.dot(shear_vec, storm_vec) / (shear_mag * storm_mag)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle_rad)
def plot_hodograph(data_file, lat, lon):
    """
    Plots a color-segmented hodograph for the given lat/lon.
    Uses RAP geopotential heights at isobaric levels (converted to MSL height),
    extracts the surface terrain elevation from the same GRIB2 file using pygrib,
    computes AGL = MSL height - surface elevation, and then interpolates the
    winds to markers at 1, 2, 3, ... 8 km AGL.
    """
    # --- 1) Load isobaric-level data using xarray/cfgrib ---
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

    u = ds_u["u"]      # Wind U (m/s)
    v = ds_v["v"]      # Wind V (m/s)
    z = ds_z["gh"]     # Geopotential (m^2/s^2)
    pressure = ds_u["isobaricInhPa"]  # Pressure levels (hPa)
    lat_isobaric = ds_u["latitude"].values
    lon_isobaric = ds_u["longitude"].values

    # --- 2) Get surface elevation using the pygrib function ---
    station_elev_m = get_elevation_pygrib(data_file, lat, lon)
    if station_elev_m is None:
        print("Surface terrain elevation not found; defaulting to 0 m.")
        station_elev_m = 0.0
    print(f"Surface terrain elevation at lat={lat:.2f}, lon={lon:.2f} is ~{station_elev_m:.1f} m")

    # --- 3) Metadata ---
    run_time = pd.to_datetime(ds_u.time.values).strftime('%Y-%m-%d %HZ')
    valid_time = pd.to_datetime(ds_u.valid_time.values).strftime('%a %Y-%m-%d %HZ')
    if lon_isobaric.min() >= 0 and lon_isobaric.max() > 180 and lon < 0:
        lon = lon + 360
        print(f"Converted input longitude to 0-360 format: {lon}°")

    # --- 4) Interpolate wind and geopotential data at desired pressure levels ---
    levels = [
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
        750, 725, 700, 675, 650, 625, 600, 575, 550, 525,
        500, 475, 450, 425, 400, 375, 350, 325, 300, 275,
        250, 225, 200, 175, 150, 125, 100
    ]

    u_interp, v_interp, z_interp = [], [], []
    for lev in tqdm(levels, desc="Interpolating Wind/Height Data", unit="level"):
        if lev not in pressure.values:
            continue
        idx = np.abs(pressure.values - lev).argmin()
        u_level = u.isel(isobaricInhPa=idx).values
        v_level = v.isel(isobaricInhPa=idx).values
        z_level = z.isel(isobaricInhPa=idx).values

        u_pt = griddata((lat_isobaric.ravel(), lon_isobaric.ravel()),
                        u_level.ravel(), (lat, lon), method='nearest')
        v_pt = griddata((lat_isobaric.ravel(), lon_isobaric.ravel()),
                        v_level.ravel(), (lat, lon), method='nearest')
        z_pt = griddata((lat_isobaric.ravel(), lon_isobaric.ravel()),
                        z_level.ravel(), (lat, lon), method='nearest')

        u_interp.append(u_pt)
        v_interp.append(v_pt)
        z_interp.append(z_pt)

    u_interp = np.array(u_interp)
    v_interp = np.array(v_interp)
    z_interp = np.array(z_interp)

    # --- 5) Compute heights ---
    # Convert geopotential (m^2/s^2) to MSL height (m)
    z_m = (z_interp / 9.80665)*10  # Divide by g to get meters, multiply by 10 as dataset is in decameters

    # Compute height above ground level (AGL)
    z_agl_m = z_m - station_elev_m  # Subtract the surface elevation
    z_agl_km = z_agl_m / 1000.0  # Convert to km

    
    for i, p in enumerate(levels):
        #print(f"Pressure: {p} hPa, Geopotential Height (m): {z_m[i]:.2f}, Station Elev: {station_elev_m:.2f}, AGL: {z_agl_km[i]:.2f} km")
    
        print(f"AGL range at this location: {z_agl_km.min():.2f} km to {z_agl_km.max():.2f} km")
    print(f"Available geopotential height range: {np.min(z_m):.2f} m to {np.max(z_m):.2f} m")

    fig, (ax_hodo, ax_barbs) = plt.subplots(
    1, 2,
    figsize=(9, 6),
    width_ratios=[3, 0.3]  # start with something reasonable
    )
    fig.subplots_adjust(wspace=0.15)  # tighter spacing between axes
    pos = ax_barbs.get_position()
    ax_barbs.set_position([pos.x0 - 0.02, pos.y0, pos.width, pos.height])  # move left by 0.02

    h = Hodograph(ax_hodo, component_range=40)
    h.add_grid(increment=10)

     # Convert to proper units
    pressure_profile = np.array(levels) * units.hPa            # Already in levels list
    u_profile = u_interp * units('m/s')
    v_profile = v_interp * units('m/s')
    height_profile = z_agl_m * units('meter')  # AGL heights in meters

    # --- Vertical Wind Profile (barbs) ---
    # Use 0–12 km wind profile (adjust as needed)
    max_height_km = 12
    idx_max = np.argmax(z_agl_km >= max_height_km) if np.any(z_agl_km >= max_height_km) else len(z_agl_km)
    # Plot barbs (convert to units matplotlib expects)
    u_barbs = u_profile[:idx_max].to('knots').magnitude
    v_barbs = v_profile[:idx_max].to('knots').magnitude
    z_barbs = z_agl_km[:idx_max]
    ax_barbs.barbs(np.zeros_like(z_barbs), z_barbs, u_barbs, v_barbs, length=6)
    ax_barbs.set_ylim(ax_barbs.get_ylim()[0] - 0.5, max_height_km)
    ax_barbs.set_xlim(-1, 1)  # narrow x-range to center barbs
    ax_barbs.set_xticks([])
    ax_barbs.set_ylabel("Height AGL (km)")
    ax_hodo.set_title("Hodograph", fontsize=12)
    ax_barbs.set_title("Wind Profile", fontsize=12)
    
   
    # Extract surface wind (0 km AGL)
    idx_surface = np.argmin(np.abs(z_agl_km - 0))  # Find nearest index to surface
  
    # Calculate storm motions
    rm, lm, mean = bunkers_storm_motion(pressure_profile, u_profile, v_profile, height_profile)

    crit_angle = calculate_critical_angle(u_profile, v_profile, z_agl_km, rm)
    print(f"Critical Angle (0–0.5 km): {crit_angle:.1f}°")

    idx_3km = np.argmin(np.abs(z_agl_km - 3))
    u_srh = u_profile[:idx_3km + 1]  # Already has units, for the area filling
    v_srh = v_profile[:idx_3km + 1]
    
    # Compute 0–3 km SRH using MetPy 1.6.3 syntax
    srh_0_3km, _, _ = storm_relative_helicity(
        z_agl_m * units.meter,
        u_profile,
        v_profile,
        3000 * units.meter,
        storm_u=rm[0],
        storm_v=rm[1]
    )
    # Convert to floats for plotting
    u_plot = u_srh.magnitude
    v_plot = v_srh.magnitude
    rm_u = rm[0].magnitude
    rm_v = rm[1].magnitude
    # Build the polygon: storm motion point + wind profile (0–3 km) + back to storm motion
    poly_u = np.concatenate([[rm_u], u_plot, [rm_u]])
    poly_v = np.concatenate([[rm_v], v_plot, [rm_v]])

    # Compute storm-relative wind vector (Surface → BRM)
    u_sfc = u_profile[idx_surface].magnitude
    v_sfc = v_profile[idx_surface].magnitude

    # Plot on hodograph
    # Plot unfilled circles for each motion
    # Convert to plain floats for plotting
    rm_u, rm_v = rm[0].magnitude, rm[1].magnitude
    lm_u, lm_v = lm[0].magnitude, lm[1].magnitude
    mean_u, mean_v = mean[0].magnitude, mean[1].magnitude
    
    # Plot storm-relative wind vector (Surface → BRM)
  
    # Convert storm motion to floats
    rm_u, rm_v = rm[0].magnitude, rm[1].magnitude

  

   # Plot the correctly aligned SRH lobe
    ax_hodo.fill(poly_u, poly_v, color='lightblue', alpha=0.4, zorder=2)

    ax_hodo.text(0.02, 0.02,
                f"0–3 km SRH: {srh_0_3km:.0f} m²/s²",
                transform=ax_hodo.transAxes,
                fontsize=10, color='navy',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
    # Find the 3 km wind
    idx_3km = np.argmin(np.abs(z_agl_km - 3))
    u_3km = u_profile[idx_3km].magnitude
    v_3km = v_profile[idx_3km].magnitude
    rm_u = rm[0].magnitude
    rm_v = rm[1].magnitude
    
    # Plot Surface → BRM (Storm-Relative Wind)
    ax_hodo.plot([u_sfc, rm_u], 
            [v_sfc, rm_v], 
            color='blue', linewidth=1.5, linestyle='-', alpha=0.8)

    # Plot BRM → 3km Wind
    ax_hodo.plot([rm_u, u_3km], 
            [rm_v, v_3km], 
            color='blue', linewidth=1.5, linestyle='-', alpha=0.8)

    # Label the segments
    ax_hodo.text((u_sfc + rm_u) / 2, 
            (v_sfc + rm_v) / 2, 
            "SRW", color='blue', fontsize=8, ha='center', va='bottom')

    ax_hodo.text((rm_u + u_3km) / 2, 
            (rm_v + v_3km) / 2, 
            "0-3km SRH", color='blue', fontsize=9, ha='center', va='bottom')
    ax_hodo.text(0.02, 0.08,
             f"Critical Angle: {crit_angle:.1f}°",
             transform=ax_hodo.transAxes,
             fontsize=9, color='darkgreen',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
# Plot hollow circles
    ax_hodo.scatter(rm_u, rm_v, facecolors='none', edgecolors='red', marker='o', s=60, zorder=10)
    ax_hodo.scatter(lm_u, lm_v, facecolors='none', edgecolors='blue', marker='o', s=60, zorder=10)
    ax_hodo.scatter(mean_u, mean_v, facecolors='none', edgecolors='green', marker='o', s=60, zorder=10)

    # Label each marker
    ax_hodo.text(rm_u + 1, rm_v, 'BRM', color='red', fontsize=9, va='center')
    ax_hodo.text(lm_u + 1, lm_v, 'BLM', color='blue', fontsize=9, va='center')
    ax_hodo.text(mean_u + 1, mean_v, '0-6km MW', color='green', fontsize=9, va='center')

    num_points = len(u_interp)
    if num_points > 1:
        cmap = plt.get_cmap("Spectral")
        norm = mcolors.Normalize(vmin=0, vmax=num_points - 1)
        for i in range(num_points - 1):
            segment_color = cmap(norm(i))
            h.plot(u_interp[i:i+2], v_interp[i:i+2],
                   color=segment_color, linewidth=2)
    elif num_points == 1:
        h.plot(u_interp, v_interp, color='blue', marker='o')

    # --- 7) Mark and label wind vectors at specific AGL heights (1-8 km) ---
    alt_markers = [1, 2, 3, 4, 5, 6, 7, 8]
        # Interpolate wind components to exact alt_markers (1 km, 2 km, etc.)
    interp_func_u = interp1d(z_agl_km, u_interp, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_func_v = interp1d(z_agl_km, v_interp, kind='linear', bounds_error=False, fill_value="extrapolate")

    for alt in alt_markers:
        if np.min(z_agl_km) <= alt <= np.max(z_agl_km):  # Ensure within available heights
            u_alt = interp_func_u(alt)
            v_alt = interp_func_v(alt)
            
            ax_hodo.scatter(u_alt, v_alt, color='black', s=30, zorder=5)
            ax_hodo.text(u_alt + 1.0, v_alt, f"{alt} km",
                    fontsize=9, color='black', ha='left', va='center', zorder=6)

    # --- 8) Annotate metadata ---
    metadata_text = (
        f"RAP {run_time}, F000\n"
        f"VALID: {valid_time}\n"
        f"AT: {lat:.2f}°N, {lon:.2f}°{'W' if lon < 0 else 'E'}"
    )
    ax_hodo.text(0.02, 0.98, metadata_text, transform=ax_hodo.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', boxstyle="round,pad=0.3"),
            color='white')

    plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide main Tk window

    data_file = filedialog.askopenfilename(
        title="Select RAP GRIB2 File",
        filetypes=[("GRIB2 files", "*.grib2")]
    )

    if data_file:
        plot_hodograph(data_file, lat=37.0842, lon=-88.5980)
    else:
        print("No file selected.")