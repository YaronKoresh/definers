@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "proj="
set /p "proj=Please choose a project: "

if "%proj%"=="" (
	echo Project is required.
	pause
	exit /B 1
)

set "compose_file=..\docker\%proj%\docker-compose.yml"
if not exist "%compose_file%" (
	echo Unknown project: %proj%
	pause
	exit /B 1
)

docker compose -f "%compose_file%" up --build --no-start -d --force-recreate

pause
exit /B %ERRORLEVEL%
