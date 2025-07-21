"""
Main entry point for DISC Assessment API
Supports both API (FastAPI) and Streamlit modes
"""

import os
import subprocess
import sys
import logging
from config import config, validate_config, check_railway_environment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    
    logger.info("🚀 Starting DISC Assessment Application...")
    
    # Validate configuration
    validate_config()
    check_railway_environment()
    
    # Get service configuration
    service_type = config.service_type
    port = config.port
    
    logger.info(f"Service Type: {service_type}")
    logger.info(f"Port: {port}")
    logger.info(f"Production Mode: {config.is_production}")
    
    if service_type.lower() == "api":
        start_api_server(port)
    else:
        start_streamlit_server(port)

def start_api_server(port):
    """Start FastAPI server"""
    logger.info("🔧 Starting FastAPI server...")
    
    try:
        import uvicorn
        from api import app
        
        # Validate required files
        required_files = ["questions.json", "disc_descriptions.json"]
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"🚨 Missing required files: {missing_files}")
            sys.exit(1)
        
        logger.info("✅ All required files found")
        logger.info(f"🌐 Starting API server on 0.0.0.0:{port}")
        
        # Start server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True,
            # Production settings
            workers=1 if config.is_production else 1,
            loop="asyncio",
            # Timeout settings
            timeout_keep_alive=30,
            timeout_graceful_shutdown=30
        )
        
    except ImportError as e:
        logger.error(f"🚨 Failed to import required modules: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"🚨 Failed to start API server: {e}")
        sys.exit(1)

def start_streamlit_server(port):
    """Start Streamlit server"""
    logger.info("🎨 Starting Streamlit server...")
    
    try:
        # Check if disc_style.py exists
        if not os.path.exists("disc_style.py"):
            logger.error("🚨 disc_style.py not found!")
            sys.exit(1)
        
        logger.info(f"🌐 Starting Streamlit server on 0.0.0.0:{port}")
        
        # Streamlit command
        cmd = [
            sys.executable, "-m", "streamlit", "run", "disc_style.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "true",
            "--server.enableXsrfProtection", "false"
        ]
        
        # Add production settings
        if config.is_production:
            cmd.extend([
                "--server.enableWebsocketCompression", "true",
                "--server.maxUploadSize", "50"
            ])
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("👋 Streamlit server stopped by user")
    except Exception as e:
        logger.error(f"🚨 Failed to start Streamlit server: {e}")
        sys.exit(1)

def health_check():
    """Simple health check function"""
    try:
        # Test configuration
        validate_config()
        
        # Test required files
        required_files = ["questions.json", "disc_descriptions.json"]
        for file in required_files:
            if not os.path.exists(file):
                return False, f"Missing file: {file}"
        
        # Test database connection if in API mode
        if config.service_type.lower() == "api":
            try:
                from config import get_database_url
                db_url = get_database_url()
                if not db_url:
                    return False, "Database URL not configured"
            except Exception as e:
                return False, f"Database configuration error: {e}"
        
        return True, "All checks passed"
        
    except Exception as e:
        return False, f"Health check failed: {e}"

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "health":
            success, message = health_check()
            logger.info(f"Health Check: {message}")
            sys.exit(0 if success else 1)
        
        elif command == "api":
            config.service_type = "api"
            main()
        
        elif command == "streamlit":
            config.service_type = "streamlit"
            main()
        
        elif command == "config":
            logger.info("📋 Configuration Information:")
            config.log_config()
            check_railway_environment()
            sys.exit(0)
        
        else:
            logger.error(f"Unknown command: {command}")
            logger.info("Available commands: health, api, streamlit, config")
            sys.exit(1)
    else:
        # Default behavior
        main()
