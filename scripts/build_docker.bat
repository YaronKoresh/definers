@echo off

cd %~dp0

set /p "proj=Please choose a project: "
docker compose -f ../docker/%proj%.yml up --no-start

pause
exit /B 0