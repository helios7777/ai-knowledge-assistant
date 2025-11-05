import subprocess
import sys

def main():
    print(" Starting AI Orchestrator Dashboard...")
    print(" Dashboard will be available at: http://localhost:8501")
    print(" API will be available at: http://localhost:8000")
    print("\nPress Ctrl+C to stop\n")
    
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])
    
    dashboard_process = subprocess.Popen([
        sys.executable, "-m", "streamlit",
        "run", "streamlit_app.py",
        "--server.port", "8501"
    ])
    
    try:
        api_process.wait()
        dashboard_process.wait()
    except KeyboardInterrupt:
        print("\n Shutting down...")
        api_process.terminate()
        dashboard_process.terminate()

if __name__ == "__main__":
    main()