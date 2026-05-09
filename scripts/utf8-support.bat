@echo off

reg add "HKCU\Console" /v VirtualTerminalLevel /t REG_DWORD /d 1 /f >nul

chcp 65001 >nul

pause
exit /B 0
