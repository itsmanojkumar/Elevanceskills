@echo off
echo Starting Unified Chatbot Platform...
if "%APP_ENTRY%"=="" (
    set APP_ENTRY=app_platform.py
)
if "%PORT%"=="" (
    set PORT=8501
)
streamlit run %APP_ENTRY% --server.address 0.0.0.0 --server.port %PORT% --server.headless true
pause
