@echo off

cd %~dp0..

set "INSTALL_GROUP="
echo.
echo Available installation groups:
echo   1. dev       - Development tools only
echo   2. audio     - Audio processing
echo   3. image     - Image processing
echo   4. video     - Video processing
echo   5. ml        - Machine learning
echo   6. nlp       - Natural language processing
echo   7. web       - Web scraping and UI
echo   8. gpu       - GPU support
echo   9. all       - All optional groups
echo   10. cuda     - CUDA acceleration (requires CUDA Toolkit)
echo.
echo You can combine groups with commas, e.g.: dev,audio,ml
echo.
set /p "INSTALL_GROUP= Enter groups to install (default: dev): "

if "%INSTALL_GROUP%"=="" set "INSTALL_GROUP=dev"

set "CUDA_INSTALLATION_GROUP="
set /p "CUDA_INSTALLATION_GROUP= Also install CUDA optional dependencies? Y or N: "

if /I "%CUDA_INSTALLATION_GROUP%"=="Y" (
    call pip install -e ".[%INSTALL_GROUP%,cuda]" --extra-index-url https://pypi.nvidia.com
) else (
    call pip install -e ".[%INSTALL_GROUP%]"
)
call poe hook

pause
exit /B 0
