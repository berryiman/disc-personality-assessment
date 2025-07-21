import os
import sys

def get_database_url():
    """Get database URL with proper Railway configuration"""
    
    # Railway PostgreSQL biasanya provide variables ini:
    database_url = os.getenv("DATABASE_URL")
    
    # Alternatif Railway variables
    if not database_url:
        # Railway sering provide variables dengan nama berbeda
        pg_host = os.getenv("PGHOST")
        pg_port = os.getenv("PGPORT", "5432")
        pg_user = os.getenv("PGUSER")
        pg_password = os.getenv("PGPASSWORD")
        pg_database = os.getenv("PGDATABASE")
        
        if all([pg_host, pg_user, pg_password, pg_database]):
            database_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
    
    # Railway private network variables
    if not database_url:
        private_url = os.getenv("DATABASE_PRIVATE_URL")
        if private_url:
            database_url = private_url
    
    # Railway public variables
    if not database_url:
        public_url = os.getenv("DATABASE_PUBLIC_URL")
        if public_url:
            database_url = public_url
    
    if not database_url:
        print("🚨 ERROR: No database connection found!")
        print("Available environment variables:")
        for key in os.environ:
            if any(term in key.upper() for term in ['DATABASE', 'POSTGRES', 'PG']):
                print(f"  {key}={os.environ[key][:50]}...")
        
        # Exit jika tidak ada database URL
        sys.exit(1)
    
    print(f"✅ Database URL found: {database_url[:50]}...")
    return database_url

def get_port():
    """Get port from Railway environment"""
    return int(os.getenv("PORT", 8000))

def get_service_type():
    """Get service type"""
    return os.getenv("SERVICE_TYPE", "streamlit")

def is_production():
    """Check if running in production (Railway)"""
    return os.getenv("RAILWAY_ENVIRONMENT") is not None
