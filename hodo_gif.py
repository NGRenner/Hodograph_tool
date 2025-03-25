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
import imageio

# Base URL for RAP data from NOMADS
NOMADS_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/"

def get_available_rap_cycles():
    """
    Reads the RAP directory for a given date (today or yesterday) and returns:
      base_url: the URL for that directory
      available: a list of tuples (cycle, forecast, file_name)
      cycles: a sorted list of unique cycle strings (e.g., "01", "06", etc.)
      date_str: the YYYYMMDD string for that directory.
    """
    now = datetime.utcnow()
    for attempt in range(2):  # Try today, then yesterday if needed
        date_str = now.strftime('%Y%m%d')
        rap_dir = f"rap.{date_str}/"
        base_url = f"{NOMADS_URL}{rap_dir}"
        response = requests.get(base_url)
        if response.status_code != 200:
            now -= timedelta(days=1)
            continue  # Try previous day
        # Find all GRIB2 files (ignoring .idx files)
        files = re.findall(r'rap\.t(\d{2})z\.awp130pgrbf(\d{2})\.grib2(?<!\.idx)', response.text)
        if not files:
            now -= timedelta(days=1)
            continue
        # Build list of tuples: (cycle, forecast, file_name)
        available = [(cycle, fxx, f"rap.t{cycle}z.awp130pgrbf{fxx}.grib2") for (cycle, fxx) in files]
        # Deduplicate in case of repeated links
        available = list(set(available))
        cycles = sorted(set([t[0] for t in available]), reverse=True)
        return base_url, available, cycles, date_str
    print("⚠️ No valid RAP .grib2 files found!")
    return None, None, None, None

def download_rap_file(file_name, base_url):
    """
    Given a file name and the base URL, check if the file exists locally;
    if not, download it.
    Returns the local file path.
    """
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
    """
    Opens the GRIB2 file using pygrib, selects the orography field,
    and returns the elevation (in meters) at the nearest grid point to
    the specified latitude and longitude.
    """
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
    """
    Calculates the critical angle between the low-level shear vector (0-1 km)
    and the storm-relative wind vector. Returns the angle in radians.
    """
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

def plot_hodograph(data_file, lat, lon, output_file=None):
    """
    Plots a color-segmented hodograph and wind profile for the given lat/lon.
    Uses RAP geopotential heights (converted to MSL height), extracts the surface
    terrain elevation via pygrib, computes AGL = MSL height - surface elevation,
    and interpolates winds to markers at 1,2,...8 km AGL.
    If output_file is provided, saves the figure as a PNG.
    """
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

    station_elev_m = get_elevation_pygrib(data_file, lat, lon)
    if station_elev_m is None:
        station_elev_m = 0.0
    print(f"Surface elevation at {lat:.2f}°N, {lon:.2f}° is ~{station_elev_m:.1f} m")

    run_time = pd.to_datetime(ds_u.time.values).strftime('%Y-%m-%d %HZ')
    valid_time = pd.to_datetime(ds_u.valid_time.values).strftime('%a %Y-%m-%d %HZ')
    if lon_isobaric.min() >= 0 and lon_isobaric.max() > 180 and lon < 0:
        lon = lon + 360

    levels = [1000,975,950,925,900,875,850,825,800,775,
              750,725,700,675,650,625,600,575,550,525,
              500,475,450,425,400,375,350,325,300,275,
              250,225,200,175,150,125,100]

    u_interp, v_interp, z_interp = [], [], []
    for lev in tqdm(levels, desc="Interpolating", unit="level"):
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

    z_m = (z_interp / 9.80665) * 10  # Convert decameters to meters
    z_agl_m = z_m - station_elev_m
    z_agl_km = z_agl_m / 1000.0

    print(f"AGL range: {z_agl_km.min():.2f} km to {z_agl_km.max():.2f} km")
    print(f"MSL height range: {np.min(z_m):.2f} m to {np.max(z_m):.2f} m")

    fig, (ax_hodo, ax_barbs) = plt.subplots(1, 2, figsize=(9,6), width_ratios=[3,0.3])
    fig.subplots_adjust(wspace=0.15)
    pos = ax_barbs.get_position()
    ax_barbs.set_position([pos.x0 - 0.02, pos.y0, pos.width, pos.height])

    h = Hodograph(ax_hodo, component_range=40)
    h.add_grid(increment=10)

    pressure_profile = np.array(levels) * units.hPa
    u_profile = u_interp * units('m/s')
    v_profile = v_interp * units('m/s')
    height_profile = z_agl_m * units('meter')

    max_height_km = 12
    idx_max = np.argmax(z_agl_km >= max_height_km) if np.any(z_agl_km >= max_height_km) else len(z_agl_km)
    u_barbs = u_profile[:idx_max].to('knots').magnitude
    v_barbs = v_profile[:idx_max].to('knots').magnitude
    z_barbs = z_agl_km[:idx_max]
    ax_barbs.barbs(np.zeros_like(z_barbs), z_barbs, u_barbs, v_barbs, length=6)
    ax_barbs.set_ylim(ax_barbs.get_ylim()[0] - 0.5, max_height_km)
    ax_barbs.set_xlim(-1,1)
    ax_barbs.set_xticks([])
    ax_barbs.set_ylabel("Height AGL (km)")
    ax_hodo.set_title("Hodograph", fontsize=12)
    ax_barbs.set_title("Wind Profile", fontsize=12)

    rm, lm, mean = bunkers_storm_motion(pressure_profile, u_profile, v_profile, height_profile)
    crit_angle = calculate_critical_angle(u_profile, v_profile, z_agl_km, rm)
    print(f"Critical Angle (0–0.5 km): {crit_angle:.1f}°")

    target_height = 3 if np.max(z_agl_km) >= 3 else np.max(z_agl_km)
    idx_target = np.argmin(np.abs(z_agl_km - target_height))
    u_srh = u_profile[:idx_target+1]
    v_srh = v_profile[:idx_target+1]
    u_plot = u_srh.magnitude
    v_plot = v_srh.magnitude
    rm_u = rm[0].magnitude
    rm_v = rm[1].magnitude
    poly_u = np.concatenate([[rm_u], u_plot, [rm_u]])
    poly_v = np.concatenate([[rm_v], v_plot, [rm_v]])
    ax_hodo.fill(poly_u, poly_v, color='lightblue', alpha=0.4, zorder=2)

    srh_val, _, _ = storm_relative_helicity(z_agl_m * units.meter, u_profile, v_profile,
                                              target_height * units.meter, storm_u=rm[0], storm_v=rm[1])
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

    alt_markers = [1,2,3,4,5,6,7,8]
    interp_func_u = interp1d(z_agl_km, u_interp, kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_func_v = interp1d(z_agl_km, v_interp, kind='linear', bounds_error=False, fill_value="extrapolate")
    for alt in alt_markers:
        if np.min(z_agl_km) <= alt <= np.max(z_agl_km):
            u_alt = interp_func_u(alt)
            v_alt = interp_func_v(alt)
            ax_hodo.scatter(u_alt, v_alt, color='black', s=30, zorder=5)
            ax_hodo.text(u_alt+1.0, v_alt, f"{alt} km",
                         fontsize=9, color='black', ha='left', va='center', zorder=6)

    metadata_text = (
        f"RAP {run_time}, F{int((pd.to_datetime(ds_u.valid_time.values)-pd.to_datetime(ds_u.time.values)).total_seconds()/3600):03d}\n"
        f"VALID: {valid_time}\n"
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
    base_url, available, cycles, date_str = get_available_rap_cycles()
    if available is None:
        exit()
    # List available cycles
    print("Available cycles:")
    for i, cycle in enumerate(cycles):
        print(f"{i}: {cycle}Z")
    cycle_index = int(input("Enter index for desired cycle: ").strip())
    chosen_cycle = cycles[cycle_index]
    # Filter available files for the chosen cycle and sort by forecast hour
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
    
    start_idx = int(input("Enter start index: ").strip())
    end_idx = int(input("Enter end index: ").strip())
    if start_idx < 0 or end_idx >= len(valid_list) or start_idx > end_idx:
        print("Invalid indices. Exiting.")
        exit()
    
    # Define location (you can modify or prompt for these)
    lat = 37.0867
    lon = -88.6041
    
    frame_files = []
    for i in range(start_idx, end_idx + 1):
        vt, fname, fcst = valid_list[i]
        print(f"Processing valid time: {vt.strftime('%Y-%m-%d %HZ')} from file {fname}")
        file_path = download_rap_file(fname, base_url)
        frame_file = f"frame_{i:03d}.png"
        plot_hodograph(file_path, lat, lon, output_file=frame_file)
        frame_files.append(frame_file)
    
    frames = []
for frame_file in frame_files:
    frames.append(imageio.imread(frame_file))
import datetime
gif_filename = f"rap_hodograph_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.gif"
imageio.mimsave(gif_filename, frames, duration=0.5)
print(f"GIF saved as {gif_filename}")
