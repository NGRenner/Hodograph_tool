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
        "regex": r'nam\.t(\d{2})z\.awp130pgrbf(\d{2})\.grib2(?<!\.idx)'
    },
    "gfs": {
        "base": "http://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/",
        "folder": "gfs",
        # For GFS, directories are like: gfs.YYYYMMDD/<cycle>/atmos/
        "subfolder": "atmos/",
        # Regex to match files like: gfs.t18z.pgrb2full.0p50.f165
        "regex": r'gfs\.t(\d{2})z\.pgrb2full\.0p50\.f(\d{3})'
    }
}

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

def calculate_critical_angle(u_profile, v_profile, z_agl_km, storm_motion, surface_threshold=0.2):
    idx_sfc = 0  
    if z_agl_km[idx_sfc] > surface_threshold:
        print(f"Warning: Lowest available AGL is {z_agl_km[idx_sfc]:.2f} km, which is above the true surface.")
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
    if shear_mag < 0.1 or storm_mag < 0.1:
        return 0.0
    cos_theta = np.dot(shear_vec, storm_vec) / (shear_mag * storm_mag)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angle_rad

def plot_hodograph(data_file, lat, lon, model, output_file=None):
    # Open datasets using cfgrib
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
    u = ds_u["u"]
    v = ds_v["v"]
    z = ds_z["gh"]
    pressure = ds_u["isobaricInhPa"]
    lat_isobaric = ds_u["latitude"].values
    lon_isobaric = ds_u["longitude"].values

    # Get surface elevation using pygrib
    station_elev_m = get_elevation_pygrib(data_file, lat, lon)
    if station_elev_m is None:
        print("Surface terrain elevation not found; defaulting to 0 m.")
        station_elev_m = 0.0
    print(f"Surface elevation at {lat:.2f}°N, {lon:.2f}° is ~{station_elev_m:.1f} m")

    rt = pd.to_datetime(ds_u.time.values)
    vt = pd.to_datetime(ds_u.valid_time.values)
    if lon_isobaric.min() >= 0 and lon_isobaric.max() > 180 and lon < 0:
        lon = lon + 360

    # Define desired pressure levels
    levels = [
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775,
        750, 725, 700, 675, 650, 625, 600, 575, 550, 525,
        500, 475, 450, 425, 400, 375, 350, 325, 300, 275,
        250, 225, 200, 175, 150, 125, 100
    ]

    # Create coordinate grids for interpolation.
    # If the latitude and longitude arrays are 1D, convert to a 2D meshgrid.
    if lat_isobaric.ndim == 1 and lon_isobaric.ndim == 1:
        lat_grid, lon_grid = np.meshgrid(lat_isobaric, lon_isobaric, indexing='ij')
    else:
        lat_grid, lon_grid = lat_isobaric, lon_isobaric

    # Initialize lists for interpolated values and levels used
    u_interp, v_interp, z_interp, levels_used = [], [], [], []
    for lev in tqdm(levels, desc="Interpolating Wind/Height Data", unit="level"):
        if lev not in pressure.values:
            continue
        idx = np.abs(pressure.values - lev).argmin()
        u_level = u.isel(isobaricInhPa=idx).values
        v_level = v.isel(isobaricInhPa=idx).values
        z_level = z.isel(isobaricInhPa=idx).values

        u_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        u_level.ravel(), (lat, lon), method='nearest')
        v_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        v_level.ravel(), (lat, lon), method='nearest')
        z_pt = griddata((lat_grid.ravel(), lon_grid.ravel()),
                        z_level.ravel(), (lat, lon), method='nearest')

        u_interp.append(u_pt)
        v_interp.append(v_pt)
        z_interp.append(z_pt)
        levels_used.append(lev)
    
    u_interp = np.array(u_interp)
    v_interp = np.array(v_interp)
    z_interp = np.array(z_interp)
    levels_used = np.array(levels_used)

    # Build pressure profile from the levels used
    pressure_profile = levels_used * units.hPa

    # Convert geopotential to MSL height (m) based on model type
    if model in ['rap', 'nam']:
        z_m = (z_interp / 9.80665) * 10  # conversion from decameters
    else:  # For GFS, adjust multiplier as needed (here *10 as an example)
        z_m = (z_interp / 9.80665) * 10

    z_agl_m = z_m - station_elev_m
    z_agl_km = z_agl_m / 1000.0

    # Sort profiles by height
    sort_idx = np.argsort(z_agl_m)
    z_m = z_m[sort_idx]
    z_agl_m = z_agl_m[sort_idx]
    z_agl_km = z_agl_km[sort_idx]
    u_interp = u_interp[sort_idx]
    v_interp = v_interp[sort_idx]
    pressure_profile = pressure_profile[sort_idx]

    # Debug prints to check vertical extent and wind profile values
    print("=== Debug: Vertical Profile ===")
    print("AGL range (km):", z_agl_km.min(), "to", z_agl_km.max())
    print("MSL height range (m):", np.min(z_m), "to", np.max(z_m))
    print("u_profile (m/s):", u_interp)
    print("v_profile (m/s):", v_interp)

    # Plot vertical wind profile (barbs)
    fig, (ax_hodo, ax_barbs) = plt.subplots(1, 2, figsize=(9,6), width_ratios=[3,0.3])
    fig.subplots_adjust(wspace=0.15)
    pos = ax_barbs.get_position()
    ax_barbs.set_position([pos.x0 - 0.02, pos.y0, pos.width, pos.height])
    h = Hodograph(ax_hodo, component_range=40)
    h.add_grid(increment=10)

    # Create wind profiles with units
    u_profile = u_interp * units('m/s')
    v_profile = v_interp * units('m/s')
    height_profile = z_agl_m * units('meter')

    # Compute storm motions using Bunkers method on the full profile
    rm, lm, mean = bunkers_storm_motion(pressure_profile, u_profile, v_profile, height_profile)
    crit_angle = calculate_critical_angle(u_profile, v_profile, z_agl_km, rm)
    print(f"Critical Angle (0–0.5 km): {crit_angle:.1f}°")

    # Plot vertical wind profile (barbs)
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

    # Set target height for SRH calculation
    target_height = 3 if np.max(z_agl_km) >= 3 else np.max(z_agl_km)
    idx_target = np.argmin(np.abs(z_agl_km - target_height))
    u_srh = u_profile[:idx_target+1]
    v_srh = v_profile[:idx_target+1]

    # Compute 0–target_height km SRH
    srh_val, _, _ = storm_relative_helicity(height_profile, u_profile, v_profile,
                                              target_height * units.meter, storm_u=rm[0], storm_v=rm[1])
    # Build polygon for SRH lobe
    u_plot = u_srh.magnitude
    v_plot = v_srh.magnitude
    rm_u = rm[0].magnitude
    rm_v = rm[1].magnitude
    poly_u = np.concatenate([[rm_u], u_plot, [rm_u]])
    poly_v = np.concatenate([[rm_v], v_plot, [rm_v]])
    ax_hodo.fill(poly_u, poly_v, color='lightblue', alpha=0.4, zorder=2)

    ax_hodo.text(0.02, 0.02,
                 f"0–{target_height:.0f} km SRH: {srh_val:.0f} m²/s²",
                 transform=ax_hodo.transAxes,
                 fontsize=10, color='navy',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    idx_surface = np.argmin(np.abs(z_agl_km - 0))
    u_sfc = u_profile[idx_surface].magnitude
    v_sfc = v_profile[idx_surface].magnitude
    idx_target_wind = np.argmin(np.abs(z_agl_km - target_height))
    u_target = u_profile[idx_target_wind].magnitude
    v_target = v_profile[idx_target_wind].magnitude
    ax_hodo.plot([u_sfc, rm[0].magnitude], [v_sfc, rm[1].magnitude],
                 color='blue', linewidth=1.5, linestyle='-', alpha=0.8)
    ax_hodo.plot([rm[0].magnitude, u_target], [rm[1].magnitude, v_target],
                 color='blue', linewidth=1.5, linestyle='-', alpha=0.8)
    ax_hodo.text((u_sfc+rm[0].magnitude)/2, (v_sfc+rm[1].magnitude)/2, "SRW",
                 color='blue', fontsize=8, ha='center', va='bottom')
    ax_hodo.text((rm[0].magnitude+u_target)/2, (rm[1].magnitude+v_target)/2,
                 f"0-{target_height:.0f}km SRH",
                 color='blue', fontsize=9, ha='center', va='bottom')
    ax_hodo.text(0.02, 0.08,
                 f"Critical Angle: {crit_angle:.1f}°",
                 transform=ax_hodo.transAxes,
                 fontsize=9, color='darkgreen',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    rm_u, rm_v = rm[0].magnitude, rm[1].magnitude
    lm_u, lm_v = lm[0].magnitude, lm[1].magnitude
    mean_u, mean_v = mean[0].magnitude, mean[1].magnitude
    ax_hodo.scatter(rm_u, rm_v, facecolors='none', edgecolors='red', marker='o', s=60, zorder=10)
    ax_hodo.scatter(lm_u, lm_v, facecolors='none', edgecolors='blue', marker='o', s=60, zorder=10)
    ax_hodo.scatter(mean_u, mean_v, facecolors='none', edgecolors='green', marker='o', s=60, zorder=10)
    ax_hodo.text(rm_u+1, rm_v, 'BRM', color='red', fontsize=9, va='center')
    ax_hodo.text(lm_u+1, lm_v, 'BLM', color='blue', fontsize=9, va='center')
    ax_hodo.text(mean_u+1, mean_v, '0-6km MW', color='green', fontsize=9, va='center')

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

    alt_markers = [1, 2, 3, 4, 5, 6, 7, 8]
    interp_func_u = interp1d(z_agl_km, u_interp, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_func_v = interp1d(z_agl_km, v_interp, kind='linear', bounds_error=False, fill_value="extrapolate")
    for alt in alt_markers:
        if np.min(z_agl_km) <= alt <= np.max(z_agl_km):
            u_alt = interp_func_u(alt)
            v_alt = interp_func_v(alt)
            ax_hodo.scatter(u_alt, v_alt, color='black', s=30, zorder=5)
            ax_hodo.text(u_alt+1.0, v_alt, f"{alt} km",
                         fontsize=9, color='black', ha='left', va='center', zorder=6)

    fcst_hour = int((vt - rt).total_seconds() // 3600)
    metadata_text = (
        f"{model.upper()} {rt.strftime('%Y-%m-%d %HZ')}, F{fcst_hour:03d}\n"
        f"VALID: {vt.strftime('%a %Y-%m-%d %HZ')}\n"
        f"AT: {lat:.2f}°N, {lon:.2f}°{'W' if lon < 0 else 'E'}"
    )
    ax_hodo.text(0.02, 0.98, metadata_text, transform=ax_hodo.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', boxstyle="round,pad=0.3"),
                 color='white')
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
    if model in ["rap", "nam"]:
        base_url, available, cycles, date_str = get_available_model_cycles(model)
        if available is None:
            exit()
        print("Available cycles:")
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
        print("Available run cycles:")
        for i, cycle in enumerate(cycles):
            print(f"{i}: {cycle}Z")
        cycle_index = int(input("Enter index for desired run cycle: ").strip())
        chosen_cycle = cycles[cycle_index]
        final_base_url, valid_list = get_available_gfs_valid_times(base_url_date, chosen_cycle)
        if valid_list is None:
            exit()
        print("Available valid times:")
        for i, (vt, fname, fcst) in enumerate(valid_list):
            print(f"{i}: {vt.strftime('%Y-%m-%d %HZ')} - File: {fname}")
        idx = int(input("Enter index for desired valid time: ").strip())
        vt, fname, fcst = valid_list[idx]
        file_path = download_model_file(fname, final_base_url)
    lat = float(input("Enter latitude: ").strip())
    lon = float(input("Enter longitude: ").strip())
    plot_hodograph(file_path, lat, lon, model)
