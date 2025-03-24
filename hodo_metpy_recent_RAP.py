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
from metpy.calc import wind_speed, wind_direction, bunkers_storm_motion
from metpy.units import units
import pygrib

# Base URL for RAP data from NOMADS
NOMADS_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/rap/prod/"

def get_latest_rap_file():
    """
    Determines the most recent RAP GRIB2 file from NOMADS, ensuring it exists before returning.
    """
    now = datetime.utcnow()
    
    for attempt in range(2):  # Try today, then yesterday if needed
        date_str = now.strftime('%Y%m%d')
        rap_dir = f"rap.{date_str}/"
        url = f"{NOMADS_URL}{rap_dir}"

        response = requests.get(url)
        if response.status_code != 200:
            now -= timedelta(days=1)  # Go back one day
            continue  # Retry with the previous day
        
        # Find all GRIB2 files (excluding .idx files)
        files = re.findall(r'rap\.t(\d{2})z\.awp130pgrbf(\d{2})\.grib2(?<!\.idx)', response.text)
        if not files:
            now -= timedelta(days=1)  # Go back one day
            continue  # Retry with the previous day

        # Sort files by latest cycle and forecast hour
        files = sorted(files, key=lambda x: (int(x[0]), int(x[1])), reverse=True)
        latest_cycle, latest_fxx = files[0]
        latest_file = f"rap.t{latest_cycle}z.awp130pgrbf{latest_fxx}.grib2"
        latest_url = f"{url}{latest_file}"

        return latest_url, latest_file

    print("⚠️ No valid RAP .grib2 files found! The data may not be available yet.")
    return None
def download_rap_file():
    """
    Checks if the latest RAP file exists locally; if not, downloads it.
    Returns the local file path.
    """
    result = get_latest_rap_file()
    if not result:
        return None

    latest_url, latest_file = result

    # Set file path relative to this script's directory.
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
    # --- 6) Plot the hodograph ---
    fig, ax = plt.subplots(figsize=(6,6))
    h = Hodograph(ax, component_range=40)
    h.add_grid(increment=10)
     # Convert to proper units
    pressure_profile = np.array(levels) * units.hPa            # Already in levels list
    u_profile = u_interp * units('m/s')
    v_profile = v_interp * units('m/s')
    height_profile = z_agl_m * units('meter')  # AGL heights in meters
    # Extract surface wind (0 km AGL)
    idx_surface = np.argmin(np.abs(z_agl_km - 0))  # Find nearest index to surface
  
    # Calculate storm motions
    rm, lm, mean = bunkers_storm_motion(pressure_profile, u_profile, v_profile, height_profile)

  
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
  
  
  
    # Find the 3 km wind
    idx_3km = np.argmin(np.abs(z_agl_km - 3))
    u_3km = u_profile[idx_3km].magnitude
    v_3km = v_profile[idx_3km].magnitude
    rm_u = rm[0].magnitude
    rm_v = rm[1].magnitude
    
    # Plot Surface → BRM (Storm-Relative Wind)
    ax.plot([u_sfc, rm_u], 
            [v_sfc, rm_v], 
            color='blue', linewidth=1.5, linestyle='-', alpha=0.8)

    # Plot BRM → 3km Wind
    ax.plot([rm_u, u_3km], 
            [rm_v, v_3km], 
            color='blue', linewidth=1.5, linestyle='-', alpha=0.8)

    # Label the segments
    ax.text((u_sfc + rm_u) / 2, 
            (v_sfc + rm_v) / 2, 
            "SRW", color='blue', fontsize=8, ha='center', va='bottom')

    ax.text((rm_u + u_3km) / 2, 
            (rm_v + v_3km) / 2, 
            "0-3km SRH", color='blue', fontsize=8, ha='center', va='bottom')
# Plot hollow circles
    ax.scatter(rm_u, rm_v, facecolors='none', edgecolors='red', marker='o', s=60, zorder=10)
    ax.scatter(lm_u, lm_v, facecolors='none', edgecolors='blue', marker='o', s=60, zorder=10)
    ax.scatter(mean_u, mean_v, facecolors='none', edgecolors='green', marker='o', s=60, zorder=10)

    # Label each marker
    ax.text(rm_u + 1, rm_v, 'BRM', color='red', fontsize=9, va='center')
    ax.text(lm_u + 1, lm_v, 'BLM', color='blue', fontsize=9, va='center')
    ax.text(mean_u + 1, mean_v, '0-6km MW', color='green', fontsize=9, va='center')

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
            
            ax.scatter(u_alt, v_alt, color='black', s=30, zorder=5)
            ax.text(u_alt + 1.0, v_alt, f"{alt} km",
                    fontsize=9, color='black', ha='left', va='center', zorder=6)

    # --- 8) Annotate metadata ---
    metadata_text = (
        f"RAP {run_time}, F000\n"
        f"VALID: {valid_time}\n"
        f"AT: {lat:.2f}°N, {lon:.2f}°{'W' if lon < 0 else 'E'}"
    )
    ax.text(0.02, 0.98, metadata_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', boxstyle="round,pad=0.3"),
            color='white')

    plt.title(f'Hodograph for {lat:.2f}, {lon:.2f}')
    plt.show()

if __name__ == "__main__":
    data_file = download_rap_file()
    if data_file:
        
        plot_hodograph(data_file, lat=46.36, lon=-96.28)
