set HF_ENDPOINT=https://hf-mirror.com
cd /d "%~dp0"
start python api_server.py
timeout /t 8
start streamlit run app_ui.py
