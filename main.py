import os
import subprocess
import sys

if __name__ == "__main__":
    service_type = os.getenv("SERVICE_TYPE", "streamlit")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting service: {service_type} on port {port}")
    
    if service_type == "api":
        # Run FastAPI with IPv6 binding for Railway private network
        import uvicorn
        from api import app
        uvicorn.run(app, host="::", port=port)  # ← IPv6 binding
    else:
        # Run Streamlit (default)
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "disc_style.py",
            "--server.port", str(port),
            "--server.address", "::",  # ← IPv6 binding juga
            "--server.headless", "true"
        ])
