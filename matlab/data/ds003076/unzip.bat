@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal enabledelayedexpansion

set UNZIP_TOOL="C:\Program Files\7-Zip\7z.exe"

for /D %%i in (sub-*) do (
    if exist "%%i\anat\*.nii.gz" (
        pushd "%%i\anat"
        for %%f in (*.nii.gz) do (
            !UNZIP_TOOL! e "%%f"
        )
        popd
    )
    
    if exist "%%i\func\*.nii.gz" (
        pushd "%%i\func"
        for %%f in (*.nii.gz) do (
            !UNZIP_TOOL! e "%%f"
        )
        popd
    )
)

echo done.
pause
