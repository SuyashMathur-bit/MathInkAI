@echo off
title Push MathInk.ai to GitHub
cd /d "%~dp0"
echo ========================================================
echo Pushing MathInk.ai to GitHub: SuyashMathur-bit/MathInkAI
echo ========================================================
echo.

git push origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================================
    echo [SUCCESS] Successfully pushed all files to GitHub!
    echo Check: https://github.com/SuyashMathur-bit/MathInkAI
    echo ========================================================
) else (
    echo ========================================================
    echo [ERROR] Push failed. If prompted, please authorize GitHub in your browser.
    echo ========================================================
)
echo.
pause
