@echo off
REM Example 06 launcher (Windows).
REM   run.bat          -> keyboard teleop + live camera
REM   run.bat camera   -> live camera only
"%~dp0..\..\.venv\Scripts\python.exe" "%~dp0run.py" %*
