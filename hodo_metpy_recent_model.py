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
        # Updated regex for NAM: pattern "nam.tHHz.awip12BB.tm00.grib2"
        "regex": r'nam\.t(\d{2})z\.awphys(\d{2})\.tm00\.grib2'
    },
    "gfs": {
        "base": "http://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/",
        "folder": "gfs",
        "subfolder": "atmos/",
        "regex": r'gfs\.t(\d{2})z\.pgrb2full\.0p50\.f(\d{3})'
    }
}

# RAP functions (same for rap and NAM)
def get_available_model_cycles(model):
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

# NAM-specific functions
def get_available_nam_dates():
    base_url = model_info["nam"]["base"]
    response = requests.get(base_url)
    if response.status_code != 200:
        print("Could not access NAM base directory.")
        return None, None
    # Extract directories in the form "nam.YYYYMMDD"
    dates = re.findall(r'nam\.(\d{8})', response.text)
    dates = sorted(set(dates), reverse=True)
    return base_url, dates

def get_available_nam_cycle(base_url, chosen_date):
    # NAM date directory is like "nam.YYYYMMDD/"
    date_folder = f"nam.{chosen_date}/"
    full_url = base_url + date_folder
    response = requests.get(full_url)
    if response.status_code != 200:
        print("Could not access NAM date folder:", full_url)
        return None, None
    # Use the NAM regex to extract cycle values from the file names
    matches = re.findall(model_info["nam"]["regex"], response.text)
    if not matches:
        print("No NAM files found in the directory.")
        return None, None
    cycles = sorted(set([m[0] for m in matches]), reverse=True)
    return full_url, cycles

def get_available_nam_valid_times(full_url, chosen_cycle, chosen_date):
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
    # Filter available files to only those with the chosen cycle
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

# GFS functions (unchanged)
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

def get_available_gfs_valid_times(base_url_date, chosen_cycle):
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
    model = input("Enter model (rap, nam, or gfs): ").strip().lower()
    if model not in model_info:
        print("Invalid model selection. Exiting.")
        exit()
    if model == "rap":
        base_url, available, cycles, date_str = get_available_model_cycles(model)
        if available is None:
            exit()
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
        print("Available valid times:")
        for i, (vt, fname, fcst) in enumerate(valid_list):
            print(f"{i}: {vt.strftime('%Y-%m-%d %HZ')} - File: {fname}")
        idx = int(input("Enter index for desired valid time: ").strip())
        vt, fname, fcst = valid_list[idx]
        file_path = download_model_file(fname, base_url)
    elif model == "nam":
        base_url, dates = get_available_nam_dates()
        if not dates:
            exit()
        print("Available NAM dates:")
        for i, d in enumerate(dates):
            print(f"{i}: nam.{d}")
        date_index = int(input("Enter index for desired date: ").strip())
        chosen_date = dates[date_index]
        full_url, cycles = get_available_nam_cycle(base_url, chosen_date)
        if not cycles:
            exit()
        print("Available NAM cycles:")
        for i, cycle in enumerate(cycles):
            print(f"{i}: {cycle}Z")
        cycle_index = int(input("Enter index for desired cycle: ").strip())
        chosen_cycle = cycles[cycle_index]
        full_url, valid_list = get_available_nam_valid_times(full_url, chosen_cycle, chosen_date)
        if valid_list is None:
            exit()
        print("Available NAM valid times:")
        for i, (vt, fname, fcst) in enumerate(valid_list):
            print(f"{i}: {vt.strftime('%Y-%m-%d %HZ')} - File: {fname}")
        idx = int(input("Enter index for desired valid time: ").strip())
        vt, fname, fcst = valid_list[idx]
        file_path = download_model_file(fname, full_url)
    elif model == "gfs":
        base_url_gfs, gfs_dates = get_available_gfs_dates()
        if not gfs_dates:
            exit()
        print("Available GFS dates:")
        for i, d in enumerate(gfs_dates):
            print(f"{i}: gfs.{d}")
        date_index = int(input("Enter index for desired date: ").strip())
        chosen_date = gfs_dates[date_index]
        base_url_date, cycles = get_available_gfs_cycle(base_url_gfs, chosen_date)
        if not cycles:
            exit()
        print("Available GFS run cycles:")
        for i, cycle in enumerate(cycles):
            print(f"{i}: {cycle}Z")
        cycle_index = int(input("Enter index for desired run cycle: ").strip())
        chosen_cycle = cycles[cycle_index]
        final_base_url, valid_list = get_available_gfs_valid_times(base_url_date, chosen_cycle)
        if valid_list is None:
            exit()
        print("Available GFS valid times:")
        for i, (vt, fname, fcst) in enumerate(valid_list):
            print(f"{i}: {vt.strftime('%Y-%m-%d %HZ')} - File: {fname}")
        idx = int(input("Enter index for desired valid time: ").strip())
        vt, fname, fcst = valid_list[idx]
        file_path = download_model_file(fname, final_base_url)
        
    lat = float(input("Enter latitude: ").strip())
    lon = float(input("Enter longitude: ").strip())
    plot_hodograph(file_path, lat, lon, model)
    input("Press Enter to exit...")
