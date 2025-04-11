"""
Configuration module for the application.
Loads environment variables and provides configuration settings.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'root'),
    'database': os.getenv('DB_NAME', 'ghandi'),
    'table': os.getenv('TABLE_NAME', 'vicidial_rdv'),
}

# Application configuration
APP_CONFIG = {
    'page_title': 'Dashboard Ghandi - Rendez-vous',
    'layout': 'wide',
    'debug': os.getenv('DEBUG', 'False').lower() in ('true', '1', 't'),
    'refresh_interval': int(os.getenv('REFRESH_INTERVAL', '60')),  # in seconds
    'theme': {
        'primary_color': '#0079FF',
        'secondary_color': '#00DFA2',
        'background_color': '#F6F8FA',
        'text_color': '#1F2328',
    },
}

# Cache configuration
CACHE_CONFIG = {
    'ttl': int(os.getenv('CACHE_TTL', '300')),  # Time to live in seconds
    'max_entries': int(os.getenv('CACHE_MAX_ENTRIES', '100')),
}

# Logging configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.getenv('LOG_FILE', None),
}
