import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import json
from datetime import datetime, timedelta
from skimage.feature import peak_local_max
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import requests
import re
import os
import tarfile
import gzip
from geopy.geocoders import Nominatim
import shutil

def snodas_url_from_date(date_str: str
                         ) -> tuple[str, str]:
    """
    Convert a date string like 'Nov 11 2025' into a SNODAS tar URL.

    Parameters:
        date_str (str): expected format Mon DD YYYY (3-letter month abbreviation).

    Returns:
        url (str): url to tar file in SNODAS archive
        filename (str): filename that is used to rename relevant snow depth
                        files extracted from the tarfile to obey common naming
                        system
    """
    # Parse the input date
    dt = datetime.strptime(date_str, "%b %d %Y")

    # Build the URL components
    year = dt.year
    month_num = f"{dt.month:02d}"
    month_txt = dt.strftime("%b")
    day = f"{dt.day:02d}"
    # Correct folder format is month_num_month_txt (eg. 11_Nov)
    month_folder = f"{month_num}_{month_txt}"

    # Final file name
    filename = f"SNODAS_unmasked_{year}{month_num}{day}"

    # Construct full URL
    url = (
        f"https://noaadata.apps.nsidc.org/NOAA/G02158/unmasked/"
        f"{year}/{month_folder}/{filename}.tar"
    )
    return url, filename

def scrape_file(url: str, 
                out_file: str
                ) -> None:
    """
    Takes URL for SNODAS archive file and downloads it from the internet

    Params:
        url (str): url of the tar file to be downloaded
        out_file (str): name of tar file being downloaded
    """
    # Use requests module to get tarfile
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_file, "wb") as f:
            # Use iter content to download file without reading it into 
            # memory all at once
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def read_header(txt_path: str
                ) -> dict:
    '''
    Reads the metadata of a snow depth file from corresponding txt file into a 
    Python dictionary to properly format snow depth map
    
    Parameters:
        txt_path (str): path to txt file with 1036 product code with meta data

    Returns:
        meta (dict): metadata from txt file
    '''
    # open as gz if needed
    if txt_path.endswith(".gz"):
        f = gzip.open(txt_path, "rt", encoding="ascii", errors="ignore")
    else:
        f = open(txt_path, "r", encoding="ascii", errors="ignore")

    # Initialize meta dictionary and read file contents into it
    meta = {}
    with f:
        for line in f:
            line = line.strip()
            # Disregard lines without key value info
            if not line or ":" not in line:
                continue
            # Split into "key: value"
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip()
            # Try to store numeric values as floats
            try:
                # Remove commas
                meta[key] = float(val.replace(",", ""))
            except ValueError:
                # Otherwise keep as string
                meta[key] = val
    return meta

def prepare_snodas_files(tar_path: str, 
                         filename: str, 
                         tmp_dir="tmp_extract", 
                         save_data=True
                         ) -> tuple[str, str]:
    """
    Extracts SNODAS tar file, finds 1036 snow depth files, extracts .dat.gz, 
    loads metadata, and stores relevant snowdepth files in local subdirectory

    Parameters:
        tar_path (str): path to downloaded tar file
        filename (str): outfile names for snow depth files (ex: SNODAS_unmasked_yyyymmdd)
        tmp_dir (str): name of subdirectory where extracted tar file contents are moved to
        save_data (bool): boolean determining whether or not extracted data is saved locally

    Returns:
        dat_path (str): path to extracted snowdepth raster file
        json_path (str): path to extracted snowdepth metadata file
    """
    # Make temporary directory for tar file contents
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # Get path to local data subdirectory
    local_data = os.path.join(os.getcwd(), "local_data")

    # Extract the tar file into temporary directory
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_dir, filter='data')

    # Find product 1036 files
    pattern_dat = re.compile(r"1036.*\.dat\.gz$", re.IGNORECASE)
    pattern_txt = re.compile(r"1036.*\.txt(\.gz)?$", re.IGNORECASE)
    snowdepth_dat = None
    snowdepth_txt = None

    for root, dirs, files in os.walk(tmp_dir):
        for f in files:
            full = os.path.join(root, f)
            if pattern_dat.search(f):
                snowdepth_dat = full
            if pattern_txt.search(f):
                snowdepth_txt = full

    # Load metadata from .txt.gz
    meta = read_header(snowdepth_txt)

    # Dump metadata into json file
    json_path = f"{filename}.json"
    with open(json_path, "w") as jf:
        json.dump(meta, jf, indent=2)

    # Decompress .dat.gz into raw .dat file
    dat_path = f"{filename}.dat"
    with gzip.open(snowdepth_dat, "rb") as f_in:
        with open(dat_path, "wb") as f_out:
            f_out.write(f_in.read())

    # Move relevant files to local_data subdirectory
    if save_data:
        dat_gz_dst = os.path.join(local_data, f"{filename}.dat.gz")
        txt_gz_dst = os.path.join(local_data, f"{filename}.txt.gz")
        shutil.move(snowdepth_dat, dat_gz_dst)
        shutil.move(snowdepth_txt, txt_gz_dst)

    # Remove temporary directories and files
    for filename in os.listdir(tmp_dir):
            file_path = os.path.join(tmp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    os.rmdir(tmp_dir)
    os.remove(tar_path)

    return dat_path, json_path

def format_meta(date_str: str, 
             bottom_left: tuple[float, float], 
             upper_right: tuple[float, float],  
             get_bounds=False
             ) -> tuple[dict, int, int, str]:
    """
    Takes input date of form Mon dd yyyy and converts it to yyyymmdd and updates corresponding 
    metadata pertaining to bounded area

    Params:
        date_str (str): date of the form Mon dd yyyy
        bottom_left (tuple[float, float]): bottom left bound of visualized snow depth in (lat, lon) fmt
        upper_right (tuple[float, float]): top right bound of visualized snow depth in (lat, lon) fmt
        get_bounds (bool): if True returns the bounding rows and columns of the snow depth np array

    Returns:
        meta (dict): json of metadata for snow depth on that day
        nrows (int): number of rows in snow depth raster
        ncols (int): number of columns in snow depth raster
        dat_path (str): path to snow depth raster
    """

    # Convert Mon dd yyyy -> yyyymmdd
    input_format = '%b %d %Y'
    output_format = '%Y%m%d'
    date = datetime.strptime(date_str, input_format).strftime(output_format)

    # Get dat path and load metadata from json file
    dat_path = rf'{os.getcwd()}\SNODAS_unmasked_{date}.dat'
    with open(rf'{os.getcwd()}\SNODAS_unmasked_{date}.json', 'r') as file:
        meta = json.load(file)
    
    nrows = int(meta["number of rows"])
    ncols = int(meta["number of columns"])

    if not get_bounds:
        return nrows, ncols, dat_path
    
    # get bounding lats and lons
    min_lat, min_lon = bottom_left #SW Corner
    max_lat, max_lon = upper_right #NE Corner

    # convert to array indices
    upper_row, upper_col = latlon_to_pixel(min_lat, max_lon, meta)
    lower_row, lower_col = latlon_to_pixel(max_lat, min_lon, meta)

    # update meta according to bounding area
    meta['minimum x-axis coordinate'] = min_lon
    meta['maximum x-axis coordinate'] = max_lon
    meta['minimum y-axis coordinate'] = min_lat
    meta['maximum y-axis coordinate'] = max_lat
    meta["number of rows"] = upper_row - lower_row
    meta["number of columns"] = upper_col - lower_col
    
    return lower_row, upper_row, lower_col, upper_col, meta

def load_array(date_str: str,
               bottom_left: tuple[float, float],
               upper_right: tuple[float, float]
               ) -> tuple[np.ndarray, dict]:
    '''
    Loads snowdepth raster data into a numpy array and returns the array and updated metadata 
    bounded according to user input bounds

    Parameters:
        date_str (str): date snowdepth data is being returned for in Mon dd yyyy fmt
        bottom_left (tuple[float, float]): bottom left bound of visualized snow depth in (lat, lon) fmt
        upper_right (tuple[float, float]): top right bound of visualized snow depth in (lat, lon) fmt

    Returns:
        arr (np.ndarray): array containing bounded snow depth raster data
        meta (dict): json of metadata for snow depth on given date
    '''
    
    # Use format meta to get array specifications
    nrows, ncols, dat_path = format_meta(date_str, bottom_left=(24.1, -130.5), upper_right=(54, -62.25))
    
    # Load binary raster data into array using np.frombuffer
    with open(dat_path, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=">i2").reshape((nrows, ncols))
    
    # Convert all values to float and all points of zero depth to np.nan
    arr = arr.astype(float)
    arr[arr == -9999] = np.nan

    # Use format meta to properly bound the array and update metadata before returning it
    lower_row, upper_row, lower_col, upper_col, meta = format_meta(date_str, bottom_left, upper_right, get_bounds=True)
    arr = arr[lower_row:upper_row, lower_col:upper_col]
    return arr, meta

def latlon_to_pixel(lat: float, 
                    lon: float, 
                    meta: dict
                    ) -> tuple[int, int]:
    """
    Convert latitude/longitude to SNODAS pixel indices using metadata.
    SNODAS uses a regular 0.008333° grid (~1 km).

    Parameters:
        lat (float): latitude of point
        lon (float): longitude of point
        meta (dict): json of metadata for snow depth on given date

    Returns:
        tuple[int, int]: corresponding index on snow depth array
    """
    xmin = meta["minimum x-axis coordinate"]
    ymax = meta["maximum y-axis coordinate"]
    dx = meta["x-axis resolution"]
    dy = meta["y-axis resolution"]
    col = (lon - xmin) / dx
    row = (ymax - lat) / dy
    return int(round(row)), int(round(col))

def pixel_to_latlon(row: int, 
                    col: int, 
                    meta: dict
                    ) -> tuple[float, float]:
    '''
    Takes the row and column of the snow depth array and returns coordinates

    Parameters:
        row (int): row in array
        col (int): column in array
        meta (dict): json of metadata for snow depth on given date

    Returns:
        lat (float): corresponding latitude to array index
        lon (float): corresponding longitude to array index
    '''
    dx = meta["x-axis resolution"]
    dy = meta["y-axis resolution"]
    lat = meta["maximum y-axis coordinate"] - row * dy
    lon = meta["minimum x-axis coordinate"] + col * dx
    return lat, lon

def snow_depth_at_point(arr: np.ndarray, 
                        meta: dict, 
                        lat: float, 
                        lon: float, 
                        row_col=False
                        ) -> float:
    """
    Reads SNODAS raster and returns snow depth at given lat/lon.

    Parameters:
        arr (np.ndarray): snow depth raster
        meta (dict): json of metadata for snow depth on given date
        lat (float): latitude of point
        lon (float): longitude of point
        row_col (bool): If True function takes row col as input instead of lat lon

    Returns:
        depth_meters (float): snow depth at specified coordinates
    """
    # Determine size of array
    nrows, ncols = np.shape(arr)

    # Convert coordinates to pixel index if needed
    if row_col == False:
        row, col = latlon_to_pixel(lat, lon, meta)
    else:
        row = lat
        col = lon

    # Ensure point is inside grid
    if not (0 <= row < nrows and 0 <= col < ncols):
        raise RuntimeError("Requested point is outside the SNODAS grid.")

    # SNODAS scale factor: meters / 1000
    depth_meters = arr[row, col] / 1000.0
    return depth_meters

def find_peaks(arr: np.ndarray, 
               meta: dict, 
               min_depth_m=0.1, 
               min_distance_miles=25
               ) -> tuple[list[tuple], list[tuple]]:
    '''
    This function finds the relative snow depth peaks in the snow depth raster
    using the peak_local_max method from the skimage.feature module. 

    Paramters:
        arr (np.ndarray): snow depth raster
        meta (dict): json of metadata for snow depth on given date
        min_depth_m (float): minimum depth to qualify as a peak
        min_distance_miles(float): minimum distance allowed between peaks

    Returns:
        peaks (list[tuple]): sorted list of all detected peak indices in array
        info (list[tuple]): list of corresponding lat/lon and depth information
    '''

    # First convert minimum distance between peaks from miles to pixels
    dx = meta["x-axis resolution"]
    miles_per_deg = 69.0
    miles_per_pixel = dx * miles_per_deg
    min_pixels = int(min_distance_miles / miles_per_pixel)

    # Convert the array from millimeters to meters
    arr_m = arr / 1000.0

    # Run peak finding algorithm
    coords = peak_local_max(
        arr,
        min_distance=min_pixels,
        threshold_abs=min_depth_m * 1000,  # convert m->mm
        exclude_border=False
    )

    peaks = []
    info = []

    # Append coordinate and depth information to corresponding lists
    for r, c in coords:
        depth_m = arr_m[r, c]
        lat = meta["maximum y-axis coordinate"] - r * meta["y-axis resolution"]
        lon = meta["minimum x-axis coordinate"] + c * meta["x-axis resolution"]
        peaks.append((r, c))
        info.append((lat, lon, depth_m))
    
    return peaks, info

def load_from_local(filename: str
                    ) -> tuple[str, str]:
    '''
    Extracts zipped snow depth data from local subdirectory

    Parameters:
        filename (str): file of interest, raster and metadata have same 
                        name but different file formats
    
    Returns:
        dat_path (str): path to extracted snowdepth raster file in CWD
        json_path (str): path to extracted snowdepth metadata file in CWD
    '''
    # Get path to local data directory
    local_dir = os.path.join(os.getcwd(), "local_data")

    # Get Paths to zipped snow depth data files
    dat_gz = os.path.join(local_dir, filename + ".dat.gz")
    json_gz = os.path.join(local_dir, filename + ".txt.gz")

    # Output extracted paths (current working directory)
    dat_path = os.path.join(os.getcwd(), filename + ".dat")
    json_path = os.path.join(os.getcwd(), filename + ".json")

    # Extract .dat
    with gzip.open(dat_gz, "rb") as f_in:
        with open(dat_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Extract .json
    meta = read_header(json_gz)
    json_path = f"{filename}.json"
    with open(json_path, "w") as jf:
        json.dump(meta, jf, indent=2)

    return dat_path, json_path
    
def plot_snowdepth_peaks(arr: np.ndarray, 
                         meta: dict, 
                         peaks: list[tuple]
                         ) -> None:
    """
    Plots SNODAS snow depth array with detected peaks overlaid, using cartopy to draw geographic borders.

    Parameters:
        arr (np.ndarray): snow depth raster
        meta (dict): json of metadata for snow depth on given date
        peaks (list[tuple]): list of detected peak indices in array
    
    Returns:
        None
    """
    # Build lat/lon grids
    nrows, ncols = arr.shape
    dx = meta["x-axis resolution"]
    dy = meta["y-axis resolution"]
    xmin = meta["minimum x-axis coordinate"]
    ymax = meta["maximum y-axis coordinate"]

    lons = xmin + np.arange(ncols) * dx
    lats = ymax - np.arange(nrows) * dy

    # Create Cartopy map
    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(
        f"{int(meta['created month'])}/{int(meta['created day'])}/{int(meta['created year'])} "
        "Snow Depth With Detected Peaks",
        fontsize=16
    )

    # Set map bounds to match data extent
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())

    reader = shpreader.Reader('countyl010g.shp')
    counties = list(reader.geometries())
    COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

    # Add features - state, county, international borders + coastline
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black', linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8)
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='black', linewidth=0.25)

    # Create custom colorbar using NERFC color chart
    depths = np.array([0, 1, 3, 8, 15, 30, 40, 80, 150, 250, 350, 500])*0.0254
    colors = ['white', 
              '#b2c0bf', 
              '#8abfc3',
              '#80a8c8', 
              '#6479c3', 
              '#4841bb', 
              '#5528bc', 
              '#701ab4', 
              '#9c25ac', 
              '#912674',
              '#843350', 
              '#7b4847']
    
    # Normalize the depths to [0,1]
    depths_norm = (depths - depths.min()) / (depths.max() - depths.min())

    # Create the colormap using LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "snowdepth_cmap",
        list(zip(depths_norm, colors)),
        N=256  # number of interpolation steps
        )
    
    # Plot snow depth
    arr_m = arr / 1000.0
    im = ax.imshow(
        arr_m,
        norm = Normalize(vmin=0, vmax=500*0.0254),
        extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        origin="upper",
        cmap=custom_cmap,
        transform=ccrs.PlateCarree()
    )

    # Plot colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7)
    cbar.set_label("Snow Depth (meters)")

    # Overlay peaks
    peak_lons = []
    peak_lats = []
    for (r, c) in peaks:
        lat = ymax - r * dy
        lon = xmin + c * dx
        peak_lats.append(lat)
        peak_lons.append(lon)
    ax.scatter(peak_lons, peak_lats, color="red", s=20, transform=ccrs.PlateCarree(), label="Peaks")
    ax.legend()

    # Overlay lat lon dotted grid lines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.6,
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 11}
    gl.ylabel_style = {"size": 11}

    plt.show()

def plot_snow_progression(start_date: str, 
                          end_date: str, 
                          coordinates: tuple[float, float], 
                          step_days=3, 
                          location_name=None
                          ) -> None:
    '''
    Plots snow progression at a certain point over a specified date range given by Mon dd yyyy format

    Parameters:
        start_date (str): starting date using Mon dd yyyy format (ex: Nov 11 2023)
        end_date (str): ending date using Mon dd yyyy format (ex: Feb 23 2024)
        coordinates (tuple[float, float]): Coordinates of point of interest in (lat, lon) format
        step_days (int): Specifies number of days between depth queries - proportional to resolution and runtime
        location_name (str): User input name of point of interest - defaults to coordinates (ex: 'Mt Washington')

    Returns:
        None
    '''

    # Parse input strings into datetime objects
    start_date = datetime.strptime(start_date, "%b %d %Y")
    end_date = datetime.strptime(end_date, "%b %d %Y")

    plt.figure()

    # Loop until we pass the end date
    current = start_date
    dates = []
    depths = []

    while current <= end_date:
        date_str=current.strftime("%b %d %Y")
        print(f'Retrieving data from {date_str}...')

        # Load files for each date to query snow depth
        try:
            dat_path, json_path = load_files(date_str)
        except Exception as e:
            print(f"!!! Failed to download {date_str} !!!: {e}")
            current += timedelta(days=step_days)
            continue
        
        # Query snow depth using snow_depth_at_point()
        arr, meta = load_array(date_str, (24.1, -130.5), (54, -62.25))
        depths.append(snow_depth_at_point(arr, meta, coordinates[0], coordinates[1])*39.3701)
        dates.append(date_str)

        # Remove files from current working directory
        os.remove(dat_path)
        os.remove(json_path)

        # Increment date
        current += timedelta(days=step_days)

    # Apply name to plot
    if len(location_name) != 0:
        plt.title(f'{location_name} Snow Depth Progression')
    else:
        plt.title(f'{coordinates} Snow Depth Progression')

    # Label x-axis with relevant dates and plot points
    step = max(1, len(dates) // 12)
    plt.plot(dates, depths, color='r')
    plt.xticks(dates[::step], rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.xlabel('Date')
    plt.ylabel('Depth in Inches')
    plt.show()

def load_from_archive(url: str, 
                      filename: str
                      ) -> tuple[str, str]:
    '''
    Loads snow depth data from SNODAS archive using specified url and outfile filename

    Parameters:
        url (str): url of the tar file to be downloaded
        filename (str): outfile names of relevant downloaded files

    Returns:
        dat_path (str): path to extracted snowdepth raster file in CWD
        json_path (str): path to extracted snowdepth metadata file in CWD
    '''
    # Initialize Filepath
    tar_file = rf"{os.getcwd()}\tmp_tar.tar"

    # Scrape the SNODAS tarfile from the archive
    scrape_file(url, 'tmp_tar.tar')

    # Download the snowdepth file to the directory
    dat_path, json_path = prepare_snodas_files(tar_file, filename)
    return dat_path, json_path

def load_files(date: str
               ) -> tuple[str, str]:
    '''
    Loads snow data files for specified date into the current working directory and saves 
    zipped files into local data directory for future quick access.

    Parameters:
        date (str): date of file to be loaded in Mon dd yyyy formate

    Returns:
        dat_path (str): path to extracted snowdepth raster file in CWD
        json_path (str): path to extracted snowdepth metadata file in CWD
    '''

    #Load files if not in current directory
    url, filename = snodas_url_from_date(date)
    local_data_path = os.path.join(os.getcwd(), "local_data")
    if os.path.exists(local_data_path):
        if f'{filename}.dat.gz' not in os.listdir(local_data_path):
            dat_path, json_path = load_from_archive(url, filename)
        else:
            dat_path, json_path = load_from_local(filename)
    else:
        os.makedirs(local_data_path)
        dat_path, json_path = load_from_archive(url, filename)
    return dat_path, json_path

def wipe_local_data() -> None:
    '''
    Deletes all files in local data directory
    '''
    local_dir = os.path.join(os.getcwd(), "local_data")
    for filename in os.listdir(local_dir):
            file_path = os.path.join(local_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

def plot_snow_map(date: str,
                  bottom_left=(24.1, -126.5),
                  upper_right=(54, -62.25)) -> None:
    
    '''
    Plots the snow depth map from specified date and bounds with the top ten peaks in
    the bounded region overlaid on the map. Additionally prints info corresponding to
    detected peaks
    
    Parameters:
        date_str (str): date snowdepth data is being plotted for in Mon dd yyyy fmt
        bottom_left (tuple[float, float]): bottom left bound of visualized snow depth in (lat, lon) fmt
        upper_right (tuple[float, float]): top right bound of visualized snow depth in (lat, lon) fmt

    Returns:
        None
    '''
    # Load files into CWD
    try:
        dat_path, json_path = load_files(date)
    except Exception:
        print(f"\n!!! Failed to fetch data from {date} !!!")
        print('Please try again with a different date')
        return

    # Load raster into numpy array and get metadata
    arr, meta = load_array(date, bottom_left, upper_right)
    peaks, info = find_peaks(arr, meta)

    if len(peaks) >= 10:
        num_peaks = 10
    else:
        num_peaks = len(peaks)

    # Print the top 10 snowdepths in the bounded region
    print(f'\n{'-'*32}\n| Deepest Detected Snow Depths |\n{'-'*32}')
    for i in range(num_peaks):
        lon, lat, depth = info[i]
        # geopy.geocoders module finds names of coordinates (very cool)
        try:
            geolocator = Nominatim(user_agent="my_reverse_geocoding_app")
            location = geolocator.reverse((lon, lat))
            print(f'{i+1}. Location: {location}\nCoordinates: ({round(lon, 2)},{round(lat,2)}) Depth: {depth}m\n')
        except Exception:
            print(f'{i+1}. ({round(lon, 2)},{round(lat,2)}): {depth}m')

    # Plot the raster with the peaks overlaid
    plot_snowdepth_peaks(arr, meta, peaks[:num_peaks])
    os.remove(dat_path)
    os.remove(json_path)

def get_single_depth(date: str,
                     coordinates: tuple[float, float],
                     location_name=None) -> None:
    '''
    Gets snow depth at a certain point on a specified date given in Mon dd yyyy format

    Parameters:
        date (str): specofied date of interest using Mon dd yyyy format (ex: Nov 11 2023)
        coordinates (tuple[float, float]): Coordinates of point of interest in (lat, lon) format
        location_name (str): User input name of point of interest - defaults to coordinates (ex: 'Mt Washington')

    Returns:
        None
    '''
    # Load file
    try:
        dat_path, json_path = load_files(date)
    except Exception:
        print(f"\n!!! Failed to fetch data from {date} !!!")
        print('Please try again with a different date')
        return
    
    # Load entire array
    bottom_left=(24.1, -126.5)
    upper_right=(54, -62.25)
    arr, meta = load_array(date, bottom_left, upper_right)

    # Get snow depth with specified parameters
    depth = snow_depth_at_point(arr, meta, coordinates[0], coordinates[1])

    # Print results
    if len(location_name) != 0:
        print(f'\nThe estimated snow depth at {location_name} on {date} is {round(depth*39.3701, 1)} inches')
    else:
        print(f'\nThe estimated snow depth at {coordinates} on {date} is {round(depth*39.3701, 1)} inches')

    # File cleanup
    os.remove(dat_path)
    os.remove(json_path)

def get_valid_date(prompt: str, 
                   fmt="%b %d %Y"
                   ) -> datetime:
    '''
    Asks user for a valid date until a valid date is input - date cannot be earlier than
    earliest unmasked SNODAS data or later than current date

    Parameters:
        prompt (str): Prompt given to user when getting date input
        fmt (str): date format

    Returns:
        date (datetime): datetime object corresponding to user input date
    '''
    # Get date bounds
    min_date = datetime.strptime('Dec 9 2009', fmt)
    max_date = datetime.today() + timedelta(days=1)

    # Use while loop to prompt user until valid date is input
    while True:
        user_input = input(prompt)
        try:
            date = datetime.strptime(user_input, fmt)
            if min_date <= date < max_date:
                return date
            # Ask user for valid day in SNODAS archive
            print('\nPlease input a date between Dec 9 2009 and the current date')
        except ValueError:
            # Ask user for proper date format if invalid date is entered
            print(f"\nInvalid date. Please use the Mon dd YYYY format (e.g., Jan 2 2024).")

def get_valid_int(prompt: str
                  ) -> int:
    '''
    Asks user for a valid integer until a valid integer is input

    Parameters:
        prompt (str): Prompt given to user when getting int input

    Returns:
        int: User input integer
    '''
    while True:
        user_input = input(prompt)
        try:
            return int(user_input)
        except ValueError:
            print("\nPlease enter a valid integer.")

def get_valid_coordinates(prompt: str
                          ) -> tuple[float, float]:
    '''
    Prompts user for valid coordinates within SNODAS range

    Parameters:
        prompt (str): prompt given to user while getting coordinates

    Returns:
        tuple[float, float]: user input coordinates
    '''
    while True:
        raw = input(prompt).strip()

        # Clean coordinate string
        if raw.startswith("(") and raw.endswith(")"):
            # remove parentheses
            raw = raw[1:-1]

        try:
            lat_str, lon_str = raw.split(",")
            lat = float(lat_str.strip())
            lon = float(lon_str.strip())
        except Exception:
            print("\nInvalid format. Please enter coordinates like (44.26, -71.30).")
            continue

        # Range checking
        if not (24.1 <= lat <= 54):
            print("\nLatitude out of bounds. Must be between 24.1 and 54.")
            continue

        if not (-126.25 <= lon <= -62.25):
            print("\nLongitude out of bounds. Must be between -126.25 and -62.25.")
            continue

        return (lat, lon)

def get_valid_letter(prompt: str, 
                     valid_letters: tuple[str]
                     ) -> str:
    '''
    Gets valid letter choice from user

    Parameters:
        prompt (str): prompt given to user while asking for letters
        valid_letters (tuple[str]): tuple of valid letters user can select

    Returns:
        str: user input letter    
    '''
    while True:
        user_input = input(prompt)
        
        if user_input in valid_letters:
            return user_input.lower()
        else:
            print(f"\nInvalid choice, please input one of the following letters: {valid_letters}")

def main() -> None:
    '''
    Executes SNODAS archive utility with user friendly interface and preset options
    '''
    exit = False
    # Use while loop to prompt user with options
    while exit == False:

        # Get initial choice
        choice = get_valid_int('\nSelect an option:\n' \
        '1. Plot snow depth map\n' \
        '2. Plot Snow depth progression\n' \
        '3. Get snow depth at a certain point\n'\
        '4. Wipe local data\n'\
        '5. Exit\n\n')

        # Snow depth map
        if choice == 1:
            bound_choice = get_valid_letter('\nSelect which area you would like to plot:\n' \
            'a: Northeast\n' \
            'b: Northwest\n' \
            'c: Rockies\n' \
            'd: Midwest\n' \
            'e: Sierra Nevada\n' \
            'f: Central Appalachia\n' \
            'g: Full Range\n\n',
            ('a', 'b', 'c', 'd', 'e', 'f', 'g'))

            date = get_valid_date('\nInput a date to plot using Mon dd yyyy format:\n\n')
            date = date.strftime("%b %d %Y")

            if bound_choice == 'a':
                plot_snow_map(date, bottom_left=(40,-80))
            elif bound_choice == 'b':
                plot_snow_map(date, bottom_left=(43.8, -126.25), upper_right=(54, -114.31))
            elif bound_choice == 'c':
                plot_snow_map(date, bottom_left=(36.23, -118.7), upper_right=(47.55, -103.51))
            elif bound_choice == 'd':
                plot_snow_map(date, bottom_left=(38.5, -102), upper_right=(49.7, -81))
            elif bound_choice == 'e':
                plot_snow_map(date, bottom_left=(35.4, -122.25), upper_right=(40, -116.17))
            elif bound_choice == 'f':
                plot_snow_map(date, bottom_left=(35.1, -85), upper_right=(41, -74.2))
            elif bound_choice == 'g':
                plot_snow_map(date)
            
            continue_choice = get_valid_letter('\nWould you like to continue? [Y/N]\n\n', ('Y', 'y', 'N', 'n'))
            if continue_choice == 'n':
                print('\nOk, goodbye!')
                break
            else:
                continue
        
        # Snow depth progression
        elif choice == 2:

            coordinates = get_valid_letter('\nInput location of interest:\n' \
            'a: Mt. Washington, NH\n' \
            'b: Mt. Mansfield, VT\n' \
            'c: Mt. Marcy, NY\n' \
            'd: Mt. Katahdin, ME\n' \
            'e: Sugarloaf Mountain, ME\n' \
            'f: Sunday River, ME\n' \
            'g: Les Monts Groulx, QC\n'
            'h: Custom coordinates\n\n',
            ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'))


            if coordinates == 'h':
                coordinates = get_valid_coordinates('\nInput the coordinates of interest:\n\n')
                location_name = input('\nWhat is the name of this location?:\n\n')
                start_date = get_valid_date('\nInput a start date to plot from (Mon dd yyyy):\n\n')
                end_date = get_valid_date('\nInput an end date to plot to (Mon dd yyyy):\n\n')

                while end_date <= start_date:
                    print("\nEnd date must be after the start date.")
                    end_date = get_valid_date('Input an end date to plot to (Mon dd yyyy):\n\n')

                start_date = start_date.strftime("%b %d %Y")
                end_date = end_date.strftime("%b %d %Y")
                step_days = get_valid_int('\nInput the number of days you would like to step by (2-4 recommended):\n\n')
                plot_snow_progression(start_date, end_date, coordinates, step_days=step_days, location_name=location_name)

            else:
                start_date = get_valid_date('\nInput a start date to plot from (Mon dd yyyy):\n\n')
                end_date = get_valid_date('\nInput an end date to plot to (Mon dd yyyy):\n\n')

                while end_date <= start_date:
                    print("\nEnd date must be after the start date.")
                    end_date = get_valid_date('Input an end date to plot to (Mon dd yyyy):\n\n')

                start_date = start_date.strftime("%b %d %Y")
                end_date = end_date.strftime("%b %d %Y")
                step_days = get_valid_int('\nInput the number of days you would like to step by (2-4 recommended):\n\n')

            if coordinates == 'a':
                plot_snow_progression(start_date, end_date, (44.26, -71.30), step_days=step_days, location_name="Mt. Washington")
            
            elif coordinates == 'b':
                plot_snow_progression(start_date, end_date, (44.54, -72.81), step_days=step_days, location_name="Mt. Mansfield")

            elif coordinates == 'c':
                plot_snow_progression(start_date, end_date, (44.112, -73.924), step_days=step_days, location_name="Mt. Marcy")
            
            elif coordinates == 'd':
                plot_snow_progression(start_date, end_date, (45.904, -68.927), step_days=step_days, location_name="Mt. Katahdin")
            
            elif coordinates == 'e':
                plot_snow_progression(start_date, end_date, (45.033, -70.316), step_days=step_days, location_name="Sugarloaf Mountain")
            
            elif coordinates == 'f':
                plot_snow_progression(start_date, end_date, (44.468, -70.881), step_days=step_days, location_name="Sunday River")

            elif coordinates == 'g':
                plot_snow_progression(start_date, end_date, (51.53, -68.075), step_days=step_days, location_name="Les Monts Groulx")

            elif coordinates == 'i':
                plot_snow_progression(start_date, end_date, (44.1, -70.20), step_days=step_days, location_name="Bates College")

            elif coordinates not in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'):
                print('\nTo make a selection, please input a, b, c, d, e, f, g, h, or i')

            continue_choice = get_valid_letter('\nWould you like to continue? [Y/N]\n\n', ('Y', 'y', 'N', 'n'))
            if continue_choice == 'n':
                print('\nOk, goodbye!')
                break
            else:
                continue
        
        # Snow depth at certain time and point
        elif choice == 3:
            coordinates = get_valid_letter('\nInput location of interest:\n' \
            'a: Mt. Washington, NH\n' \
            'b: Mt. Mansfield, VT\n' \
            'c: Mt. Marcy, NY\n' \
            'd: Mt. Katahdin, ME\n' \
            'e: Sugarloaf Mountain, ME\n' \
            'f: Sunday River, ME\n' \
            'g: Les Monts Groulx, QC\n' \
            'h: Bates College, ME\n' \
            'i: Custom coordinates\n\n',
            ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'))

            if coordinates == 'i':
                coordinates = get_valid_coordinates('\nInput the coordinates of interest:\n\n')
                location_name = input('\nWhat is the name of this location?:\n\n')
                date = get_valid_date('\nInput a date to get snow depth from using Mon dd yyyy format:\n\n')
                date = date.strftime("%b %d %Y")
                get_single_depth(date, coordinates, location_name=location_name)

            else:
                date = get_valid_date('\nInput a date to get snow depth from using Mon dd yyyy format:\n\n')
                date = date.strftime("%b %d %Y")

            if coordinates == 'a':
                get_single_depth(date, (44.26, -71.30), location_name="Mt. Washington")
            
            elif coordinates == 'b':
                get_single_depth(date, (44.54, -72.81), location_name="Mt. Mansfield")

            elif coordinates == 'c':
                get_single_depth(date, (44.112, -73.924), location_name="Mt. Marcy")
            
            elif coordinates == 'd':
                get_single_depth(date, (45.904, -68.927), location_name="Mt. Katahdin")
            
            elif coordinates == 'e':
                get_single_depth(date, (45.033, -70.316), location_name="Sugarloaf Mountain")
            
            elif coordinates == 'f':
                get_single_depth(date, (44.468, -70.881), location_name="Sunday River")

            elif coordinates == 'g':
                get_single_depth(date, (51.53, -68.075), location_name="Les Monts Groulx")

            elif coordinates == 'h':
                get_single_depth(date, (44.1, -70.20), location_name="Bates College")
            
            elif coordinates not in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'):
                print('\nTo make a selection, please input a, b, c, d, e, f, g, h, or i')

            continue_choice = get_valid_letter('\nWould you like to continue? [Y/N]\n\n', ('Y', 'y', 'N', 'n'))
            if continue_choice == 'n':
                print('\nOk, goodbye!')
                break
            else:
                continue

        # Wipe local data
        elif choice == 4:
            wipe_choice = get_valid_letter('\nAre you sure you want to wipe ALL local data? [Y/N]\n\n', ('Y', 'y', 'N', 'n'))
            
            if wipe_choice == 'y':
                wipe_local_data()
            else:
                print('\nAborting local data deletion')
            continue

        # Exit program
        elif choice == 5:
            print('\nGoodbye!\n')
            exit = True
        
        # Loop through if number 1-5 is not selected
        elif 5 < choice or choice < 0:
            print('Please input an integer from 1 to 5')

if __name__ == '__main__':
    main()