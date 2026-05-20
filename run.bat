@echo off
chcp 65001 >/dev/null
cd /d "%~dp0"

echo ============================================
echo   MechAgent-LM
echo ============================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Run:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [INFO] Starting Streamlit with venv python...
echo.
python -m streamlit run app.py --server.port 8501

pause
