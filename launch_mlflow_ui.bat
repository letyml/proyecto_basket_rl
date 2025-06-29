
@echo off
echo Iniciando MLflow UI...

REM Obtener la ruta absoluta de la carpeta mlruns relativa a este archivo
set "MLRUNS_PATH=%~dp0mlruns"

python -m mlflow ui ^
  --backend-store-uri "file:///%MLRUNS_PATH%"

pause
