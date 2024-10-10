# Download all tifs and csvs in top level directory
rclone --drive-shared-with-me copy "Google Drive - personal":"2023 Screening CliP library/Camera Data (tif, xpim, csv)" . -v --filter "+ /*.csv" --filter "- /*/**" --filter "- /*.xpim"
rclone --drive-shared-with-me copy "Google Drive - personal":"2023 Screening CliP library/Camera Data (tif, xpim, csv)" . -v --filter "+ /*.tif" --filter "- /*/**" --filter "- /*.xpim"

# Write out total number of files
echo "Download complete."
echo "Total number of files: $(ls *.tif *.csv | wc -l)"
echo "Total number of tifs: $(ls *.tif | wc -l)"

# Check that each csv has a corresponding tiff with same filename
for csv in *.csv; do
  tif=$(echo $csv | sed 's/csv/tif/')
  if [ ! -f $tif ]; then
    echo "Missing tif file for $csv"
  fi
done

# Check that each tif has a corresponding csv with same filename
for tif in *.tif; do
  csv=$(echo $tif | sed 's/tif/csv/')
  if [ ! -f "$csv" ]; then
    echo "Missing csv file for $tif"
  fi
done

