rclone --drive-shared-with-me ls "Google Drive - personal":"2023 Screening CliP library/Camera Data (tif, xpim, csv)" --include "*.csv"
rclone --drive-shared-with-me ls "Google Drive - personal":"2023 Screening CliP library/Camera Data (tif, xpim, csv)" --include "*.tiff"

# Write out total number of files
echo "Total number of files: $(ls *.tiff *.csv | wc -l)"

# Check that each csv has a corresponding tiff with same filename
for csv in *.csv; do
  tiff=$(echo $csv | sed 's/csv/tiff/')
  if [ ! -f $tiff ]; then
    echo "Missing tiff file for $csv"
  fi
done

# Check that each tiff has a corresponding csv with same filename
for tiff in *.tiff; do
  csv=$(echo $tiff | sed 's/tiff/csv/')
  if [ ! -f $csv ]; then
    echo "Missing csv file for $tiff"
  fi
done
