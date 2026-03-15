@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo [+] Stopping all running containers...
set "has_containers="
for /f %%i in ('docker ps -aq 2^>nul') do (
	set "has_containers=1"
	docker stop %%i
)
if not defined has_containers echo [i] No containers found.

echo.
echo [+] Removing all containers...
for /f %%i in ('docker ps -aq 2^>nul') do docker rm %%i

echo.
echo [+] Removing all images...
set "has_images="
for /f %%i in ('docker images -aq 2^>nul') do (
	set "has_images=1"
	docker rmi -f %%i
)
if not defined has_images echo [i] No images found.

pause
exit /B %ERRORLEVEL%