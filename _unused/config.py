"""
Configuration module for DISC Assessment API on Railway
Handles environment variables and database configuration
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the application"""
    
    def __init__(self):
        self.database_url = self._get_database_url()
        self.port = self._get_port()
        self.service_type = self._get_service_type()
        self.is_production = self._is_production()
        self.ssl_mode = self._get_ssl_mode()
        
    def _get_database_url(self):
        """
        Get database URL dengan multiple fallbacks untuk Railway
        Railway bisa provide DATABASE_URL dalam berbagai format
        """
        logger.info("🔍 Searching for database configuration...")
        
        # Method 1: Standard DATABASE_URL
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            logger.info("✅ Found DATABASE_URL")
            return database_url
        
        # Method 2: Railway PostgreSQL service variables
        pg_host = os.getenv("PGHOST")
        pg_port = os.getenv("PGPORT", "5432")
        pg_user = os.getenv("PGUSER") 
        pg_password = os.getenv("PGPASSWORD")
        pg_database = os.getenv("PGDATABASE")
        
        if all([pg_host, pg_user, pg_password, pg_database]):
            database_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
            logger.info("✅ Built DATABASE_URL from PG* variables")
            return database_url
        
        # Method 3: Railway private network URL
        private_url = os.getenv("DATABASE_PRIVATE_URL")
        if private_url:
            logger.info("✅ Found DATABASE_PRIVATE_URL")
            return private_url
        
        # Method 4: Railway public URL
        public_url = os.getenv("DATABASE_PUBLIC_URL")
        if public_url:
            logger.info("✅ Found DATABASE_PUBLIC_URL")
            return public_url
        
        # Method 5: Alternative naming patterns
        for env_name in ["POSTGRES_URL", "POSTGRESQL_URL", "DATABASE_CONNECTION_STRING"]:
            url = os.getenv(env_name)
            if url:
                logger.info(f"✅ Found {env_name}")
                return url
        
        # Method 6: Railway specific patterns
        railway_vars = [
            "RAILWAY_DATABASE_URL",
            "RAILWAY_POSTGRES_URL", 
            "RAILWAY_DB_URL"
        ]
        for var in railway_vars:
            url = os.getenv(var)
            if url:
                logger.info(f"✅ Found {var}")
                return url
        
        # Log available environment variables for debugging
        logger.error("🚨 No database URL found!")
        self._log_available_env_vars()
        
        return None
    
    def _log_available_env_vars(self):
        """Log environment variables for debugging"""
        logger.info("Available environment variables:")
        relevant_vars = []
        
        for key in sorted(os.environ.keys()):
            if any(term in key.upper() for term in ['DATABASE', 'POSTGRES', 'PG', 'RAILWAY', 'DB']):
                value = os.environ[key]
                # Mask sensitive data
                if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                    value = "***MASKED***"
                elif len(value) > 100:
                    value = value[:50] + "..." + value[-20:]
                relevant_vars.append(f"  {key}={value}")
        
        if relevant_vars:
            for var in relevant_vars:
                logger.info(var)
        else:
            logger.info("  No relevant environment variables found")
    
    def _get_port(self):
        """Get port from Railway environment"""
        return int(os.getenv("PORT", 8000))
    
    def _get_service_type(self):
        """Get service type"""
        return os.getenv("SERVICE_TYPE", "api")
    
    def _is_production(self):
        """Check if running in production (Railway)"""
        return any([
            os.getenv("RAILWAY_ENVIRONMENT"),
            os.getenv("RAILWAY_PROJECT_ID"),
            os.getenv("RAILWAY_SERVICE_ID"),
            os.getenv("RAILWAY_DEPLOYMENT_ID"),
            os.getenv("RAILWAY_ENVIRONMENT_NAME")
        ])
    
    def _get_ssl_mode(self):
        """Get appropriate SSL mode for database connection"""
        if self.is_production:
            return "require"
        return "prefer"
    
    def validate(self):
        """Validate configuration"""
        if not self.database_url:
            error_msg = "Database URL not found in environment variables"
            logger.error(f"🚨 {error_msg}")
            
            if self.is_production:
                logger.error("🚨 In production mode, database is required!")
                return False
            else:
                logger.warning("⚠️ In development mode, will use fallback database")
                self.database_url = "postgresql://user:password@localhost:5432/dbname"
        
        return True
    
    def get_masked_database_url(self):
        """Get database URL with masked password for logging"""
        if not self.database_url:
            return None
            
        masked_url = self.database_url
        if "@" in masked_url:
            parts = masked_url.split("@")
            if ":" in parts[0]:
                auth_parts = parts[0].split(":")
                if len(auth_parts) >= 3:  # postgresql://user:password
                    auth_parts[-1] = "***"
                    parts[0] = ":".join(auth_parts)
                    masked_url = "@".join(parts)
        return masked_url
    
    def log_config(self):
        """Log current configuration"""
        logger.info("📋 Current configuration:")
        logger.info(f"   Port: {self.port}")
        logger.info(f"   Service Type: {self.service_type}")
        logger.info(f"   Production Mode: {self.is_production}")
        logger.info(f"   SSL Mode: {self.ssl_mode}")
        
        masked_url = self.get_masked_database_url()
        if masked_url:
            logger.info(f"   Database URL: {masked_url}")
        else:
            logger.error("   Database URL: NOT CONFIGURED")

# Global configuration instance
config = Config()

# Convenience functions for backward compatibility
def get_database_url():
    """Get database URL"""
    return config.database_url

def get_port():
    """Get port from Railway environment"""
    return config.port

def get_service_type():
    """Get service type"""
    return config.service_type

def is_production():
    """Check if running in production (Railway)"""
    return config.is_production

def get_ssl_mode():
    """Get SSL mode for database connection"""
    return config.ssl_mode

def validate_config():
    """Validate configuration and exit if invalid in production"""
    if not config.validate():
        if config.is_production:
            logger.error("🚨 Configuration validation failed in production!")
            sys.exit(1)
        else:
            logger.warning("⚠️ Configuration validation failed, but continuing in development mode")
    
    config.log_config()
    return True

# Environment variable checker
def check_railway_environment():
    """Check if running in Railway environment and log details"""
    railway_vars = {
        'RAILWAY_ENVIRONMENT': os.getenv('RAILWAY_ENVIRONMENT'),
        'RAILWAY_PROJECT_ID': os.getenv('RAILWAY_PROJECT_ID'),
        'RAILWAY_SERVICE_ID': os.getenv('RAILWAY_SERVICE_ID'),
        'RAILWAY_DEPLOYMENT_ID': os.getenv('RAILWAY_DEPLOYMENT_ID'),
        'RAILWAY_SERVICE_NAME': os.getenv('RAILWAY_SERVICE_NAME'),
        'RAILWAY_ENVIRONMENT_NAME': os.getenv('RAILWAY_ENVIRONMENT_NAME')
    }
    
    railway_detected = any(railway_vars.values())
    
    if railway_detected:
        logger.info("🚂 Railway environment detected:")
        for key, value in railway_vars.items():
            if value:
                # Mask sensitive IDs
                if len(value) > 20:
                    display_value = value[:8] + "..." + value[-4:]
                else:
                    display_value = value
                logger.info(f"   {key}: {display_value}")
    else:
        logger.info("💻 Local development environment detected")
    
    return railway_detected

# Database connection parameters
def get_db_connection_params():
    """Get database connection parameters for psycopg2"""
    if not config.database_url:
        return None
    
    params = {
        'dsn': config.database_url,
        'sslmode': config.ssl_mode,
        'connect_timeout': 30,
        'application_name': 'disc-assessment-api'
    }
    
    # Add additional parameters for Railway
    if config.is_production:
        params.update({
            'keepalives_idle': 600,
            'keepalives_interval': 30,
            'keepalives_count': 3
        })
    
    return params

# Initialize configuration on import
if __name__ != "__main__":
    # Auto-validate when imported
    validate_config()
    check_railway_environment()
