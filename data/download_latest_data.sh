#!/bin/bash

# define valid flags
OPT_SPEC="hxs:d:"

# initialize flags
EXECUTE=false
SOURCE_DIR=""
DEST_DIR=""

# function to print the help message
function print_help() {
  echo "Usage: download_latest_data.sh -[x]d <directory>"
  echo "Download the latest data from the Google Drive shared folder"
  echo "Options:"
  echo "  -h              Display this help message"
  echo "  -s <directory>  Specify the rclone source directory for downloading the files from Google Drive"
  echo "  -d <directory>  Specify the destination directory for the downloaded files"
  echo "  -x              Execute the download (optional, will only print the files to be downloaded if not set)"
}


# parse the flags
while getopts $OPT_SPEC opt; do
  case $opt in
    h|help)
      print_help
      exit 0
      ;;
    x)
      EXECUTE=true
      ;;
    s)
      SOURCE_DIR=$OPTARG
      ;;
    d)
      DEST_DIR=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      print_help >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      print_help >&2
      exit 1
      ;;
  esac
done

# Check if the required source and destinations are provided
if [ -z "${SOURCE_DIR}" ]; then
    echo "Error: The -s option is required" >&2
    print_help >&2
    exit 1
elif [ -z "${DEST_DIR}" ]; then
    echo "Error: The -d option is required" >&2
    print_help >&2
    exit 1
fi

# print out the files that would be copied if the -x flag is not set
if [ "$EXECUTE" = false ]; then
  echo "The following files would be downloaded to ${DEST_DIR}:"
  # download all the csv and tif files in the immediate directory (not recursive)
  rclone --drive-shared-with-me ls "${SOURCE_DIR}" --include "*.csv" --include "*.tif" --max-depth 1
  exit 0
fi

# download the files
rclone --drive-shared-with-me copy "${SOURCE_DIR}" "${DEST_DIR}" --include "*.csv" --include "*.tif" --max-depth 1

# Write out total number of files
echo "Download complete."
# save the current directory
CURRENT_DIR=$(pwd)
cd $DEST_DIR
echo "Total number of files: $(ls *.tif *.csv | wc -l)"

# Check that each csv has a corresponding tiff with same filename in the destination directory
for csv in *.csv; do
  tif=$(echo $csv | sed 's/csv/tif/')
  if [ ! -f $tif ]; then
    echo "Missing tif file for $csv"
  fi
done

# Check that each tif has a corresponding csv with same filename
for tif in *.tif; do
  csv=$(echo $tif | sed 's/tif/csv/')
  if [ ! -f $csv ]; then
    echo "Missing csv file for $tif"
  fi
done

# return to the original directory
cd $CURRENT_DIR