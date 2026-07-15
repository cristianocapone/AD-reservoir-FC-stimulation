@echo off
REM Windows wrapper to run the BIDS conversion helper using the local bids_env environment when available.
setlocal
set LOCAL_PYTHON=%~dp0bids_env\python.exe
set LOCAL_DCM2NIIX=%~dp0bids_env\Library\bin\dcm2niix.exe
if exist "%LOCAL_PYTHON%" (
  if exist "%LOCAL_DCM2NIIX%" (
    "%LOCAL_PYTHON%" "%~dp0convert_to_bids.py" --dcm2niix "%LOCAL_DCM2NIIX%" %*
    exit /b %ERRORLEVEL%
  )
)
REM Fallback to the global Python interpreter if local environment is not available.
set PYTHON_PATH=C:\ProgramData\anaconda3\python.exe
if not exist "%PYTHON_PATH%" (
  echo Local bids_env environment not found and global Python not found at %PYTHON_PATH%.
  echo Please install Python or create the local environment with conda.
  exit /b 1
)
"%PYTHON_PATH%" "%~dp0convert_to_bids.py" %*
