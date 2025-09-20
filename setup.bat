@echo off
echo Creating Python virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install -r requirements.txt

echo Setting up frontend...
cd frontend
echo Installing Node.js dependencies...
npm install

echo Setup complete! You can now:
echo 1. Start the backend: python backend/main.py
echo 2. Start the frontend: cd frontend ^& npm start