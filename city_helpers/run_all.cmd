@echo off
setlocal
call F:\NYCU\Miniforge3\condabin\conda.bat activate sem_graphs
python -B F:\NYCU\city_helpers\repo_guard.py snapshot F:\NYCU\city_experiments\repository_before.json
python -B F:\NYCU\city_helpers\sweep_city.py --mode all
python -B F:\NYCU\city_helpers\summarize_results.py
for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Import-Csv 'F:\NYCU\city_experiments\best_result.csv').checkpoint"`) do set BEST_CHECKPOINT=%%i
python -B F:\NYCU\city_helpers\measure_latency.py --checkpoint "%BEST_CHECKPOINT%"
python -B F:\NYCU\city_helpers\repo_guard.py verify F:\NYCU\city_experiments\repository_before.json
endlocal
