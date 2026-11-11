@echo off
REM Voice moderation bot launcher.
REM   - double-click to run it in a window
REM   - or let Task Scheduler run it at logon
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist logs mkdir logs

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set STAMP=%DT:~0,8%-%DT:~8,6%
set LOGFILE=logs\bot-%STAMP%.log

REM Python's logging writes to stderr, so merge both streams into one file.
".venv\Scripts\python.exe" -u bot.py > "%LOGFILE%" 2>&1
set CODE=%ERRORLEVEL%

REM Only hold the window open if it fell over, so the scheduler isn't left hanging.
if not "%CODE%"=="0" (
  echo Bot exited with code %CODE%. See %LOGFILE%
  if defined SESSIONNAME pause
)
