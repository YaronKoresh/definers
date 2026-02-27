@echo off

cd %~dp0

set /p "proj=Please choose a project: "

docker compose -f ../docker/%proj%/docker-compose.yml up --build --no-start -d --force-recreate

pause
exit /B 0
