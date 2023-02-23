##################################################################################
##                               Import packages                                ##
##################################################################################
import numpy as np
import pandas as pd
import xarray as xr
import os
from pathlib import Path
import argparse
import sys
import datetime as dt
import copy
import glob
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polygonize
import scipy.interpolate as interp
import concurrent.futures

##################################################################################
##                               Some functions                                 ##
##################################################################################


def convert_noaa_updn_ds_to_subset_df(up_ds, dn_ds):
    """
    Specific for NOAA AOML gliders, convert xarray dataset to pandas dataframe and subset to just the variables needed
    :up_df: xarray dataset of the upcast
    :dn_df: xarray dataset of the downcast
    :returns: merged upcast downcast pandas dataframe
    """
    ##################################################################################
    ## Convert to dataframe
    dn_df = dn_ds.to_dataframe()
    ## Keep track of if it is an up or downcast
    dn_df['downs'] = 1
    ## Subset to just the variables needed
    dn_df = dn_df[["ctd_time","ctd_pressure","aanderaa4831_dissolved_oxygen","downs"]]
    ## Change index
    dn_df = dn_df.set_index('ctd_time')

    ##################################################################################
    ## Convert to dataframe
    up_df = up_ds.to_dataframe()
    ## Keep track of if it is an up or downcast
    up_df['downs'] = 0
    ## Subset to just the variables needed
    up_df = up_df[["ctd_time","ctd_pressure","aanderaa4831_dissolved_oxygen","downs"]]
    ## Change index
    up_df = up_df.set_index('ctd_time')

    ## Append the two dataframes together
    df = dn_df.append(up_df)

    return df



def convert_noaa_updn_ds_to_one_df(up_ds, dn_ds):
    """
    Specific for NOAA AOML gliders, convert xarray dataset to pandas dataframe and subset to just the variables needed
    :up_df: xarray dataset of the upcast
    :dn_df: xarray dataset of the downcast
    :returns: merged upcast downcast pandas dataframe
    """
    ##################################################################################
    ## Convert to dataframe
    dn_df = dn_ds.to_dataframe()
    ## Keep track of if it is an up or downcast
    dn_df['downs'] = 1
    ## Subset to just the variables needed
    dn_df = dn_df[["ctd_time","ctd_pressure","longitude","latitude","temperature","salinity",
                   "density","aanderaa4831_dissolved_oxygen","du","dv","su","sv","downs"]]
    ## Change index
    dn_df = dn_df.set_index('ctd_time')

    ##################################################################################
    ## Convert to dataframe
    up_df = up_ds.to_dataframe()
    ## Keep track of if it is an up or downcast
    up_df['downs'] = 0
    ## Subset to just the variables needed
    up_df = up_df[["ctd_time","ctd_pressure","longitude","latitude","temperature","salinity",
                   "density","aanderaa4831_dissolved_oxygen","du","dv","su","sv","downs"]]
    ## Change index
    up_df = up_df.set_index('ctd_time')

    ## Append the two dataframes together
    df = dn_df.append(up_df)

    return df


def apply_time_shift(df, varname, shift_seconds, merge_original=True):
    """
    Apply a specified time shift to a variable.
    :param df: pandas dataframe containing the variable of interest (varname), pressure, and time as the index
    :param varname: sensor variable name (e.g. dissolved_oxygen)
    :param shift_seconds: desired time shift in seconds
    :param merge_original: merge shifted dataframe with the original dataframe, default is False
    :returns: pandas dataframe containing the time-shifted variable, pressure, and time as the index
    """
    # split off the variable and profile direction identifiers into a separate dataframe
    try:
        sdf = pd.DataFrame(dict(shifted_var=df[varname],
                                downs=df['downs']))
    except KeyError:
        sdf = pd.DataFrame(dict(shifted_var=df[varname]))

    # calculate the shifted timestamps
    tm_shift = df.index - dt.timedelta(seconds=shift_seconds)

    # append the shifted timestamps to the new dataframe and drop the original time index
    sdf['time_shift'] = tm_shift
    sdf.reset_index(drop=True, inplace=True)

    # rename the new columns and set the shifted timestamps as the index
    sdf = sdf.rename(columns={'time_shift': 'ctd_time',
                              'downs': 'downs_shifted'})
    sdf = sdf.set_index('ctd_time')

    if merge_original:
        # merge back into the original dataframe and drop rows with nans
        df2 = df.merge(sdf, how='outer', left_index=True, right_index=True)

        # drop the original variable
        df2.drop(columns=[varname, 'downs'], inplace=True)
        df2 = df2.rename(columns={'shifted_var': f'{varname}_shifted',
                                  'downs_shifted': 'downs'})
    else:
        df2 = sdf.rename(columns={'shifted_var': f'{varname}_shifted',
                                  'downs_shifted': 'downs'})

    return df2



def calculate_pressure_range(df):
    """
    Calculate pressure range for a dataframe
    :param df: pandas dataframe containing pressure
    :returns: pressure range
    """
    min_pressure = np.nanmin(df.ctd_pressure)
    max_pressure = np.nanmax(df.ctd_pressure)

    return max_pressure - min_pressure


def identify_nans(dataset, varname):
    # identify where not nan
    non_nan_ind = np.invert(np.isnan(dataset[varname].values))
    # get locations of non-nans
    non_nan_i = np.where(non_nan_ind)[0]

    # identify where pressure is not nan
    press_non_nan_ind = np.where(np.invert(np.isnan(dataset.ctd_pressure.values)))[0]

    return non_nan_i, press_non_nan_ind


def interp_pressure(df):
    """
    Linear interpolate pressure in a time-shifted dataframe.
    :param df: pandas dataframe containing pressure and the time-shifted data, and time as the index
    :returns: pandas dataframe containing the time-shifted variable, interpolated pressure, and time as the index
    """
    # drop the original time index
    df['ctd_pressure'] = df['ctd_pressure'].interpolate(method='linear', limit_direction='both')

    return df




def pressure_bins(df, interval=0.25):
    """
    Bin data according to a specified depth interval, calculate median values for each bin.
    :param df: pandas dataframe containing pressure and the time-shifted data, and time as the index
    :param interval: optional pressure interval for binning, default is 0.25
    :returns: pandas dataframe containing depth-binned median data
    """
    # specify the bin intervals
    max_pressure = np.nanmax(df.ctd_pressure)
    bins = np.arange(0, max_pressure, interval).tolist()
    bins.append(bins[-1] + interval)

    # calculate the bin for each row
    df['bin'] = pd.cut(df['ctd_pressure'], bins)

    # calculate depth-binned median
    # used median instead of mean to account for potential unreasonable values not removed by QC
    df = df.groupby('bin').median()

    return df



def calc_area_after_shift(trajectory_resample, max_seconds):

    # define shifts in seconds to test
    shifts = np.arange(0, max_seconds, 1).tolist()
    shifts.append(max_seconds)

    # For each shift, shift the master dataframes by x seconds, bin data by 0.25 dbar,
    # calculate area between curves
    areas = []
    for shift in shifts:

        trajectory_shift = apply_time_shift(trajectory_resample, 'aanderaa4831_dissolved_oxygen', shift)
        trajectory_interp = interp_pressure(trajectory_shift)
        trajectory_interp.dropna(subset=['aanderaa4831_dissolved_oxygen_shifted'], inplace=True)

        # find down identifiers that were averaged in the resampling and reset
        downs = np.array(trajectory_interp['downs'])
        ind = np.argwhere(downs == 0.5).flatten()
        downs[ind] = downs[ind - 1]
        trajectory_interp['downs'] = downs

        # after shifting and interpolating pressure, divide df into down and up profiles
        downs_df = trajectory_interp[trajectory_interp['downs'] == 1].copy()
        ups_df = trajectory_interp[trajectory_interp['downs'] == 0].copy()

        # don't calculate area if a down or up profile group is missing
        if np.logical_or(len(downs_df) == 0, len(ups_df) == 0):
            area = np.nan
        else:
            # check the pressure range
            downs_pressure_range = calculate_pressure_range(downs_df)
            ups_pressure_range = calculate_pressure_range(ups_df)

            # don't calculate area if either profile grouping spans <3 dbar
            if np.logical_or(downs_pressure_range < 3, ups_pressure_range < 3):
                area = np.nan
            else:
                # bin data frames
                downs_binned = pressure_bins(downs_df)
                downs_binned.dropna(inplace=True)
                ups_binned = pressure_bins(ups_df)
                ups_binned.dropna(inplace=True)

                downs_ups = downs_binned.append(ups_binned.iloc[::-1])

                # calculate area between curves
                polygon_points = downs_ups.values.tolist()
                polygon_points.append(polygon_points[0])
                polygon = Polygon(polygon_points)
                polygon_lines = polygon.exterior
                polygon_crossovers = polygon_lines.intersection(polygon_lines)
                polygons = polygonize(polygon_crossovers)
                valid_polygons = MultiPolygon(polygons)
                area = valid_polygons.area

        areas.append(area)
    
    return areas



def process_oxygen(data_path):

    deployment_list = sorted(glob.glob(''.join([data_path,'*.nc'])))

    ## Pull out just the ### part of the file name
    fnames = []
    for x in np.arange(0,len(deployment_list)):
    #for x in np.arange(0,1):
        fnum = deployment_list[x][-15:-11]
        fnames.append(fnum)

    ## There's two of these so just give me one
    fnames = np.unique(fnames)

    default_shift  = 30
    segment_shifts = np.empty(len(fnames))
    segment_shifts[:] = np.nan

    ## Now pull out the full file path
    for ind in np.arange(0,len(fnames)):

        dn_file = glob.glob(''.join([data_path,"*",fnames[ind],"_dn_AOML.nc"]))[0]
        up_file = glob.glob(''.join([data_path,"*",fnames[ind],"_up_AOML.nc"]))[0]

        ## Load data
        dn_ds = xr.open_dataset(dn_file)
        up_ds = xr.open_dataset(up_file)

        ## Only proceed if there is oxygen data AND there's both up and down cast data
        if ('aanderaa4831_dissolved_oxygen' in list(dn_ds.keys())) & ('aanderaa4831_dissolved_oxygen' in list(up_ds.keys())) & (len(dn_ds.ctd_time) > 1) & (len(up_ds.ctd_time) > 1):

            ## Make a MASTER dataframe with everything for saving afterwards
            tot_df = convert_noaa_updn_ds_to_one_df(up_ds, dn_ds)
            ## Subset and convert dataframes
            df = convert_noaa_updn_ds_to_subset_df(up_ds, dn_ds)
            ## removes duplicates and syncs the dataframes so they can be merged when shifted
            trajectory_resample = df.resample('1s').mean()

            max_seconds = 90

            ## Test shifting curves by 0-60 seconds
            areas = calc_area_after_shift(trajectory_resample, max_seconds)

            ##### Now calculate what the optimal shift is!

            # if >50% of the values are nan, return nan
            fraction_nan = np.sum(np.isnan(areas)) / len(areas)
            if fraction_nan > .5:
                shift_val = np.nan
            else:
                # find the shift that results in the minimum area between the curves
                opt_shift = int(np.nanargmin(areas))

                # if the optimal shift is zero or last shift tested (couldn't find a minimal
                # area within the times tested), use the closest non-nan shift from the
                # previous segments
                if np.logical_or(opt_shift == 0, opt_shift == np.nanmax(max_seconds)):
                    non_nans = ~np.isnan(segment_shifts[ind])
                    try:
                        opt_shift = int(segment_shifts[ind][non_nans][-1])
                    except IndexError:
                        # if there are no previous non-nan optimal shifts, use the default
                        # value from the config file
                        opt_shift = default_shift

                segment_shifts[ind] = opt_shift

            # Now do the shift
            trajectory_shifted = apply_time_shift(trajectory_resample, 'aanderaa4831_dissolved_oxygen', segment_shifts[ind])
            trajectory_shifted_interp = interp_pressure(trajectory_shifted)
            trajectory_shifted_interp.dropna(subset=['aanderaa4831_dissolved_oxygen_shifted'], inplace=True)

            ## Interp total df on the ctd_time of the shifted df
            tot_df_interp = tot_df.reindex(tot_df.index.union(trajectory_shifted_interp.index)).interpolate(method='time').reindex(trajectory_shifted_interp.index)
            ## Merge
            df_merged = pd.merge(trajectory_shifted_interp, tot_df_interp, left_index=True, right_index=True)
            ## Drop columns we don't need
            df_merged = df_merged.drop(['ctd_pressure_y','downs_y'],axis=1)
            ## Rename columns
            df_merged = df_merged.rename(columns={"ctd_pressure_x": "ctd_pressure","downs_x":"downs"})
            ## Add the optimal time shift used here
            df_merged['time_shift'] = opt_shift

            #####################################
            ## Now save output
            ## Make directory if there is none
            directory_name = str(up_ds.trajectory.values)[2:-1]
            save_path = ''.join(['/home/jg1200/data/AOML_Data/Processed/',directory_name])

            Path(save_path).mkdir(parents=True, exist_ok=True)

            ## Save file
            #save_file_name = save_path+"/"+directory_name+"-"+fnames[ind]+".csv"
            save_file_name = ''.join([save_path,"/",directory_name,"-",fnames[ind],".csv"])
            df_merged.to_csv(save_file_name)  

            now = dt.datetime.now()
            print('Done:',data_path,'Index:',ind,"out of",len(fnames),'at',now)

            ## For saving memory
            dn_ds.close()
            up_ds.close()
            del dn_ds, up_ds, tot_df, df, trajectory_resample, areas, trajectory_shifted, trajectory_shifted_interp, tot_df_interp, df_merged
        
        else:
            now = dt.datetime.now()
            print('No Oxygen Data:',data_path,'Index:',ind,"out of",len(fnames),'at',now)
            os.remove(dn_file)
            os.remove(up_file)

        


##################################################################################
##                                Run the code!                                 ##
##################################################################################
## Paths to data
total_deployment_list = glob.glob("/home/jg1200/data/AOML_Data/Raw/*/*/", recursive = True)

## Test for a single deployment
#process_oxygen(total_deployment_list[28])

## Run the process oxygen data code in parallel for all the deployments (so fast, thanks Mike Smith)
with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
    executor.map(process_oxygen, total_deployment_list)
