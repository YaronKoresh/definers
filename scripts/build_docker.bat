@echo off

cd %~dp0

set /p "proj=Please choose a project: "

echo [+] Stopping all running containers...
for /f %%i in ('docker ps -aq') do docker stop %%i

echo.
echo [+] Removing all containers...
for /f %%i in ('docker ps -aq') do docker rm %%i

echo.
echo [+] Removing all images...
for /f %%i in ('docker images -aq') do docker rmi -f %%i

docker compose -f ../docker/%proj%.yml up --no-start

pause
exit /B 0