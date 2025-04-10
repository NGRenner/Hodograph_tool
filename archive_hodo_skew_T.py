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
import tkinter as tk
from tkinter import filedialog

# Model‑specific information for NOMADS (RAP, NAM, and GFS)
model_info = {
    "rap": {
        "base": "http://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/",
        "folder": "rap",
        "regex": r'rap\.t(\d{2})z\.awp130pgrbf(\d{2})\.grib2(?<!\.idx)'
    },
    "nam": {
        "base": "http://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/",
        "folder": "nam",
        "regex": r'nam\.t(\d{2})z\.awp130pgrbf(\d{2})\.grib2(?<!\.idx)'
    },
    "gfs": {
        "base": "http://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/",
        "folder": "gfs",
        "subfolder": "atmos/",
        "regex": r'gfs\.t(\d{2})z\.pgrb2full\.0p50\.f(\d{3})'
    }
}

def get_latest_rap_file():
    now = datetime.utcnow()
    for attempt in range(2):  # Try today, then yesterday if needed
        date_str = now.strftime('%Y%m%d')
        rap_dir = f"rap.{date_str}/"
        url = f"{model_info['rap']['base']}{rap_dir}"
        response = requests.get(url)
        if response.status_code != 200:
            now -= timedelta(days=1)
            continue
        files = list(set(re.findall(model_info['rap']['regex'], response.text)))
        if not files:
            now -= timedelta(days=1)
            continue
        available = [(cycle, fxx, f"rap.t{cycle}z.awp130pgrbf{fxx}.grib2") for (cycle, fxx) in files]
        cycles = sorted(set([t[0] for t in available]), reverse=True)
        latest_cycle = cycles[0]
        files_for_latest = sorted([t for t in available if t[0] == latest_cycle], key=lambda t: int(t[1]))
        base_date = datetime.strptime(date_str, "%Y%m%d")
        run_time = base_date + timedelta(hours=int(latest_cycle))
        valid_list = []
        for t in files_for_latest:
            fxx = t[1]
            valid_time = run_time + timedelta(hours=int(fxx))
            valid_list.append((valid_time, t[2]))
        print(f"Available valid times for RAP run (cycle {latest_cycle}Z) on {date_str}:")
        for i, (vt, fname) in enumerate(valid_list):
            print(f"{i}: {vt.strftime('%Y-%m-%d %HZ')}  -  File: {fname}")
        index = input("Enter the index of the desired valid time: ").strip()
        try:
            index = int(index)
            if 0 <= index < len(valid_list):
                selected_file = valid_list[index][1]
                latest_url = f"{url}{selected_file}"
                return latest_url, selected_file
            else:
                print("Invalid index. Exiting.")
                return None
        except ValueError:
            print("Invalid input. Exiting.")
            return None
    print("⚠️ No valid RAP .grib2 files found! The data may not be available yet.")
    return None

def download_rap_file():
    result = get_latest_rap_file()
    if not result:
        return None
    latest_url, latest_file = result
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, latest_file)
    if os.path.exists(file_path):
        print(f"✅ File already exists: {file_path}")
        return file_path
    print(f"Downloading: {latest_url}")
    response = requests.get(latest_url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as f, tqdm(
            desc="Downloading RAP File", total=total_size, unit='B', unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"✅ Downloaded RAP data: {file_path}")
        return file_path
    else:
        print("❌ Failed to download RAP file.")
        return None

def get_elevation_pygrib(file_path, lat_target, lon_target):
    grbs = pygrib.open(file_path)
    orog_msgs = grbs.select(shortName='orog', typeOfLevel='surface')
    if len(orog_msgs) == 0:
        raise ValueError("No orography field found in the file.")
    grb = orog_msgs[0]
    print("Using GRIB message for orography:")
    print(grb)
    lats, lons = grb.latlons()
    elevations = grb.values
    if lons.min() >= 0 and lons.max() > 180 and lon_target < 0:
        lon_target = lon_target + 360
    dist = np.sqrt((lats - lat_target)**2 + (lons - lon_target)**2)
    min_index = np.unravel_index(np.argmin(dist), dist.shape)
    return elevations[min_index]

def calculate_critical_angle(u_profile, v_profile, z_agl_km, storm_motion):
    idx_sfc = np.argmin(np.abs(z_agl_km - 0))
    u_sfc = u_profile[idx_sfc].to('m/s').magnitude
    v_sfc = v_profile[idx_sfc].to('m/s').magnitude
    idx_1km = np.argmin(np.abs(z_agl_km - 1))
    u_1km = u_profile[idx_1km].to('m/s').magnitude
    v_1km = v_profile[idx_1km].to('m/s').magnitude
    shear_u = u_1km - u_sfc
    shear_v = v_1km - v_sfc
    rm_u = storm_motion[0].to('m/s').magnitude
    rm_v = storm_motion[1].to('m/s').magnitude
    storm_rel_u = rm_u - u_sfc
    storm_rel_v = rm_v - v_sfc
    shear_vec = np.array([shear_u, shear_v])
    storm_vec = np.array([storm_rel_u, storm_rel_v])
    shear_mag = np.linalg.norm(shear_vec)
    storm_mag = np.linalg.norm(storm_vec)
    if shear_mag == 0 or storm_mag == 0:
        return np.nan
    cos_theta = np.dot(shear_vec, storm_vec) / (shear_mag * storm_mag)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angle_rad

def plot_hodograph(data_file, lat, lon, model="rap"):
    ds_u = xr.open_dataset(
        data_file, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'u'}}, indexpath=None)
    ds_v = xr.open_dataset(
        data_file, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'v'}}, indexpath=None)
    ds_z = xr.open_dataset(
        data_file, engine='cfgrib',
        backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh'}}, indexpath=None)

    u = ds_u["u"]
    v = ds_v["v"]
    z = ds_z["gh"]
    pressure = ds_u["isobaricInhPa"]
    lat_isobaric = ds_u["latitude"].values
    lon_isobaric = ds_u["longitude"].values

    # --- Get surface elevation ---
    station_elev_m = get_elevation_pygrib(data_file, lat, lon)
    if station_elev_m is None:
        print("Surface terrain elevation not found; defaulting to 0 m.")
        station_elev_m = 0.0
    print(f"Surface terrain elevation at lat={lat:.2f}, lon={lon:.2f} is ~{station_elev_m:.1f} m")

    rt = pd.to_datetime(ds_u.time.values).strftime('%Y-%m-%d %HZ')
    valid_time = pd.to_datetime(ds_u.valid_time.values).strftime('%a %Y-%m-%d %HZ')
    if lon_isobaric.min() >= 0 and lon_isobaric.max() > 180 and lon < 0:
        lon += 360
        print(f"Converted input longitude to 0-360 format: {lon}°")

    # --- Interpolate over desired pressure levels ---
    levels = [1000,975,950,925,900,875,850,825,800,775,
              750,725,700,675,650,625,600,575,550,525,
              500,475,450,425,400,375,350,325,300,275,
              250,225,200,175,150,125,100]
    u_interp, v_interp, z_interp, levels_used = [], [], [], []
    if lat_isobaric.ndim == 1 and lon_isobaric.ndim == 1:
        lat_grid, lon_grid = np.meshgrid(lat_isobaric, lon_isobaric, indexing='ij')
    else:
        lat_grid, lon_grid = lat_isobaric, lon_isobaric
    for lev in tqdm(levels, desc="Interpolating Wind/Height Data"):
        if lev not in pressure.values:
            continue
        idx = np.abs(pressure.values - lev).argmin()
        u_level = u.isel(isobaricInhPa=idx).values
        v_level = v.isel(isobaricInhPa=idx).values
        z_level = z.isel(isobaricInhPa=idx).values

        u_pt = griddata((lat_grid.ravel(), lon_grid.ravel()), u_level.ravel(), (lat, lon), method='nearest')
        v_pt = griddata((lat_grid.ravel(), lon_grid.ravel()), v_level.ravel(), (lat, lon), method='nearest')
        z_pt = griddata((lat_grid.ravel(), lon_grid.ravel()), z_level.ravel(), (lat, lon), method='nearest')

        u_interp.append(u_pt)
        v_interp.append(v_pt)
        z_interp.append(z_pt)
        levels_used.append(lev)

    u_interp = np.array(u_interp)
    v_interp = np.array(v_interp)
    z_interp = np.array(z_interp)
    levels_used = np.array(levels_used)

    # --- Compute heights ---
    # For RAP, geopotential is in decameters
    z_m = (z_interp / 9.80665) * 10  
    z_agl_m = z_m - station_elev_m
    z_agl_km = z_agl_m / 1000.0

    # Remove levels below ground (if any)
    valid = z_agl_m >= 0
    u_interp = u_interp[valid]
    v_interp = v_interp[valid]
    z_interp = z_interp[valid]
    levels_used = levels_used[valid]

    z_m = (z_interp / 9.80665) * 10
    z_agl_m = z_m - station_elev_m
    z_agl_km = z_agl_m / 1000.0

    # Sort vertical arrays (ascending height)
    sort_idx = np.argsort(z_agl_m)
    u_interp = u_interp[sort_idx]
    v_interp = v_interp[sort_idx]
    z_agl_m = z_agl_m[sort_idx]
    z_agl_km = z_agl_km[sort_idx]
    pressure_profile = levels_used[sort_idx] * units.hPa

    # Convert interpolated arrays to proper units
    u_profile = u_interp * units('m/s')
    v_profile = v_interp * units('m/s')
    height_profile = z_agl_m * units.meter

    # --- Storm Motion and Critical Angle ---
    rm, lm, mean = bunkers_storm_motion(pressure_profile, u_profile, v_profile, height_profile)
    crit_angle = calculate_critical_angle(u_profile, v_profile, z_agl_km, rm)
    print(f"Critical Angle (0–0.5 km): {crit_angle:.1f}°")

    # --- Plotting ---
    fig, (ax_hodo, ax_barbs) = plt.subplots(1, 2, figsize=(9, 6), width_ratios=[3, 0.3])
    fig.subplots_adjust(wspace=0.15)
    pos = ax_barbs.get_position()
    ax_barbs.set_position([pos.x0 - 0.02, pos.y0, pos.width, pos.height])
    h = Hodograph(ax_hodo, component_range=40)
    h.add_grid(increment=10)

    # Barbs
    max_height_km = 12
    idx_max = np.argmax(z_agl_km >= max_height_km) if np.any(z_agl_km >= max_height_km) else len(z_agl_km)
    u_barbs = u_profile[:idx_max].to('knots').magnitude
    v_barbs = v_profile[:idx_max].to('knots').magnitude
    z_barbs = z_agl_km[:idx_max]
    ax_barbs.barbs(np.zeros_like(z_barbs), z_barbs, u_barbs, v_barbs, length=6)
    ax_barbs.set_ylim(ax_barbs.get_ylim()[0] - 0.5, max_height_km)
    ax_barbs.set_xlim(-1, 1)
    ax_barbs.set_xticks([])
    ax_barbs.set_ylabel("Height AGL (km)")
    ax_hodo.set_title("Hodograph", fontsize=12)
    ax_barbs.set_title("Wind Profile", fontsize=12)

    # SRH Calculation (0–3 km)
    idx_3km = np.argmin(np.abs(z_agl_km - 3))
    u_srh = u_profile[:idx_3km + 1]
    v_srh = v_profile[:idx_3km + 1]
    srh_val, _, _ = storm_relative_helicity(height_profile, u_profile, v_profile, 3000 * units.meter,
                                              storm_u=rm[0], storm_v=rm[1])
    # Plot SRH polygon
    u_knots = u_profile.to('knots').magnitude
    v_knots = v_profile.to('knots').magnitude
    # Insert a point at the surface (0 km) if needed
    u_0 = float((interp1d(z_agl_km, u_knots, bounds_error=False, fill_value="extrapolate"))(0))
    v_0 = float((interp1d(z_agl_km, v_knots, bounds_error=False, fill_value="extrapolate"))(0))
    u_knots = np.insert(u_knots, 0, u_0)
    v_knots = np.insert(v_knots, 0, v_0)
    z_knots = np.insert(z_agl_km, 0, 0.0)
    rm_u = rm[0].to('knots').magnitude
    rm_v = rm[1].to('knots').magnitude
    u_srh_plot = u_knots[:idx_3km + 1]
    v_srh_plot = v_knots[:idx_3km + 1]
    poly_u = np.concatenate([[rm_u], u_srh_plot, [rm_u]])
    poly_v = np.concatenate([[rm_v], v_srh_plot, [rm_v]])
    ax_hodo.fill(poly_u, poly_v, color='lightblue', alpha=0.4, zorder=2)
    ax_hodo.text(0.02, 0.02,
                f"0–3 km SRH: {srh_val:.0f} m²/s²",
                transform=ax_hodo.transAxes,
                fontsize=10, color='navy',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Plot SRW line
    idx_surface = np.argmin(np.abs(z_agl_km - 0))
    u_sfc = u_profile[idx_surface].to('knots').magnitude
    v_sfc = v_profile[idx_surface].to('knots').magnitude
    idx_target_wind = np.argmin(np.abs(z_agl_km - 3))
    u_target = u_profile[idx_target_wind].to('knots').magnitude
    v_target = v_profile[idx_target_wind].to('knots').magnitude
    ax_hodo.plot([u_sfc, rm_u], [v_sfc, rm_v], color='blue', linewidth=1.5, linestyle='-', alpha=0.8)
    ax_hodo.plot([rm_u, u_target], [rm_v, v_target], color='blue', linewidth=1.5, linestyle='-', alpha=0.8)
    ax_hodo.text((u_sfc+rm_u)/2, (v_sfc+rm_v)/2, "SRW", color='blue', fontsize=8, ha='center', va='bottom')
    ax_hodo.text((rm_u+u_target)/2, (rm_v+v_target)/2, "0-3km SRH", color='blue', fontsize=9, ha='center', va='bottom')
    ax_hodo.text(0.02, 0.08,
                f"Critical Angle: {crit_angle:.1f}°",
                transform=ax_hodo.transAxes,
                fontsize=9, color='darkgreen',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Storm motion markers
    lm_u, lm_v = lm[0].to('knots').magnitude, lm[1].to('knots').magnitude
    mean_u, mean_v = mean[0].to('knots').magnitude, mean[1].to('knots').magnitude
    ax_hodo.scatter(rm_u, rm_v, facecolors='none', edgecolors='red', marker='o', s=60, zorder=10)
    ax_hodo.scatter(lm_u, lm_v, facecolors='none', edgecolors='blue', marker='o', s=60, zorder=10)
    ax_hodo.scatter(mean_u, mean_v, facecolors='none', edgecolors='green', marker='o', s=60, zorder=10)
    ax_hodo.text(rm_u+1, rm_v, 'BRM', color='red', fontsize=9, va='center')
    ax_hodo.text(lm_u+1, lm_v, 'BLM', color='blue', fontsize=9, va='center')
    ax_hodo.text(mean_u+1, mean_v, '0-6km MW', color='green', fontsize=9, va='center')

    # Plot profile line using a color map
    num_points = len(u_knots)
    if num_points > 1:
        cmap = plt.get_cmap("Spectral")
        norm = mcolors.Normalize(vmin=0, vmax=num_points - 1)
        for i in range(num_points - 1):
            segment_color = cmap(norm(i))
            h.plot(u_knots[i:i+2], v_knots[i:i+2], color=segment_color, linewidth=2)
    elif num_points == 1:
        h.plot(u_knots, v_knots, color='blue', marker='o')

    # Altitude markers
    interp_func_u = interp1d(z_agl_km, u_knots, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_func_v = interp1d(z_agl_km, v_knots, kind='linear', bounds_error=False, fill_value="extrapolate")
    for alt in range(1, 9):
        if np.min(z_agl_km) <= alt <= np.max(z_agl_km):
            ua = float(interp_func_u(alt))
            va = float(interp_func_v(alt))
            ax_hodo.scatter(ua, va, color='black', s=30, zorder=5)
            ax_hodo.text(ua + 1.0, va, f"{alt} km", fontsize=9, color='black', ha='left', va='center', zorder=6)

    metadata_text = (
        f"RAP {rt}, F{int((pd.to_datetime(ds_u.valid_time.values)-pd.to_datetime(ds_u.time.values)).total_seconds()//3600):03d}\n"
        f"VALID: {valid_time}\n"
        f"AT: {lat:.2f}°N, {lon:.2f}°{'W' if lon < 0 else 'E'}"
    )
    ax_hodo.text(0.02, 0.98, metadata_text, transform=ax_hodo.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', boxstyle="round,pad=0.3"),
                 color='white')

    plt.show()

if __name__ == "__main__":
    # Instead of automatically downloading a file, use a file explorer
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    data_file = filedialog.askopenfilename(title="Select RAP GRIB2 File", filetypes=[("GRIB2 files", "*.grb2")])
    if not data_file:
        print("No file selected. Exiting.")
        exit()
    lat = float(input("Enter latitude: ").strip())
    lon = float(input("Enter longitude: ").strip())
    # For RAP only, specify model="rap"
    plot_hodograph(data_file, lat, lon, model="rap")
    input("Press Enter to exit...")
