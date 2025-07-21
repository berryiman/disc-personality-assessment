import os
import subprocess
import sys

if __name__ == "__main__":
    service_type = os.getenv("SERVICE_TYPE", "streamlit")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting service: {service_type} on port {port}")
    
    if service_type == "api":
        # Run FastAPI with IPv4 binding (for external access)
        import uvicorn
        from api import app
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Run Streamlit with IPv4 binding (for external access)
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "disc_style.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0",  # ← IPv4 untuk external access
            "--server.headless", "true"
        ])
