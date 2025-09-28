@echo off

cd %~dp0

set "tag_id=latest"
set /p "image_name=Please write the project name: "

for /f "delims=" %%i in ('docker images --filter "reference=yaronkoresh/%image_name%" --format "{{.ID}}"') do (
    set "image_id=%%i"
)

docker tag %image_id% yaronkoresh/%image_name%:%tag_id%
docker push yaronkoresh/%image_name%:%tag_id%

pause
exit /B 0