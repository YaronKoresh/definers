@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "tag_id=latest"
set "image_name="
set /p "image_name=Please choose a project: "

if "%image_name%"=="" (
    echo Project is required.
    pause
    exit /B 1
)

for /f "delims=" %%i in ('docker images --filter "reference=yaronkoresh/%image_name%" --format "{{.ID}}"') do (
    set "image_id=%%i"
)

if not defined image_id (
    echo No local image found for yaronkoresh/%image_name%.
    pause
    exit /B 1
)

docker tag !image_id! yaronkoresh/%image_name%:%tag_id%
docker push yaronkoresh/%image_name%:%tag_id%

pause
exit /B %ERRORLEVEL%