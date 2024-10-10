# Import Rclone - this absolute path is hardcoded to the shared laptop
$rclonePath = "C:\Users\Burlacot lab\Downloads\rclone-v1.68.1-windows-amd64\rclone-v1.68.1-windows-amd64\rclone.exe"

# Download all tifs and csvs in top-level directory
& $rclonePath --drive-shared-with-me copy '"Google Drive - personal":"2023 Screening CliP library/Camera Data (tif, xpim, csv)"' . -v --filter "+ /*.csv" --filter "- /*/**" --filter "- /*.xpim"
& $rclonePath --drive-shared-with-me copy '"Google Drive - personal":"2023 Screening CliP library/Camera Data (tif, xpim, csv)"' . -v --filter "+ /*.tif" --filter "- /*/**" --filter "- /*.xpim"

# Download latest copy of identity spreadsheet
& $rclonePath --drive-shared-with-me copy '"Google Drive - personal":"2023 Screening CliP library/Identities of Strains on Plates/Finalized Identities Phase I plates.xlsx"' . -vv --update

# Write out total number of files
Write-Output "Download complete."
$totalFiles = (Get-ChildItem -Filter *.tif, *.csv).Count
$totalTifs = (Get-ChildItem -Filter *.tif).Count

Write-Output "Total number of files: $totalFiles"
Write-Output "Total number of tifs: $totalTifs"

# Delete these random tifs
#Remove-Item -Filter "Copy of*"

# Check that each csv has a corresponding tif with the same filename
foreach ($csv in Get-ChildItem -Filter *.csv) {
    $tif = $csv.Name -replace '\.csv$', '.tif'
    if (-not (Test-Path -Path $tif)) {
        Write-Output "Missing tif file for $($csv.Name)"
    }
}

# Check that each tif has a corresponding csv with the same filename
foreach ($tif in Get-ChildItem -Filter *.tif) {
    $csv = $tif.Name -replace '\.tif$', '.csv'
    if (-not (Test-Path -Path $csv)) {
        Write-Output "Missing csv file for $($tif.Name)"
    }
}
