"""
Central configuration management for AI Literature Meta-Analysis Engine
All project settings, API endpoints, and parameters are defined here.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Main configuration class for the Literature Analysis Engine"""
    
    # =============================================================================
    # API ENDPOINTS & SETTINGS
    # =============================================================================
    
    # ArXiv API Configuration
    ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
    ARXIV_DELAY = 3  # seconds between requests (respect rate limits)
    
    # PubMed API Configuration  
    PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    PUBMED_DELAY = 1  # seconds between requests
    
    # Semantic Scholar API Configuration
    SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/"
    SEMANTIC_SCHOLAR_DELAY = 1  # seconds between requests
    
    # =============================================================================
    # AUTHENTICATION & SECURITY
    # =============================================================================
    
    # Email required for PubMed API (NCBI requirement)
    EMAIL = os.getenv('EMAIL', 'your.email@example.com')
    
    # Optional API keys (for enhanced rate limits)
    SEMANTIC_SCHOLAR_API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY', None)
    
    # =============================================================================
    # DATA COLLECTION PARAMETERS
    # =============================================================================
    
    # Default limits for paper collection
    MAX_PAPERS_DEFAULT = 100
    MAX_PAPERS_TEST = 20      # For testing/development
    MAX_PAPERS_PRODUCTION = 1000  # For full analysis
    
    # Available data sources
    AVAILABLE_SOURCES = ['arxiv', 'pubmed', 'semantic_scholar']
    
    # =============================================================================
    # FILE PATHS & STORAGE
    # =============================================================================
    
    # Data directories
    DATA_DIR = "data"
    RAW_DATA_DIR = f"{DATA_DIR}/raw"
    PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
    SAMPLE_DATA_DIR = f"{DATA_DIR}/sample"
    
    # Database configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///literature_analysis.db')
    
    # =============================================================================
    # ANALYSIS PARAMETERS
    # =============================================================================
    
    # NLP Settings
    MIN_SIMILARITY_THRESHOLD = 0.7
    CLUSTERING_EPS = 0.3
    CLUSTERING_MIN_SAMPLES = 2
    
    # Trend analysis settings
    TREND_ANALYSIS_MIN_PAPERS = 5
    TREND_ANALYSIS_TIME_WINDOW = 24  # months
    
    # Research gap detection
    GAP_DETECTION_KEYWORDS = [
        'future work', 'further research', 'limitation', 
        'not addressed', 'remains unclear', 'requires investigation'
    ]
    
    # =============================================================================
    # MODEL CONFIGURATION
    # =============================================================================
    
    # Pre-trained models
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
    SPACY_MODEL = "en_core_web_sm"
    
    # =============================================================================
    # WEB APPLICATION SETTINGS
    # =============================================================================
    
    # Streamlit configuration
    STREAMLIT_PORT = 8501
    STREAMLIT_HOST = "localhost"
    
    # Page configuration
    PAGE_TITLE = "AI Literature Meta-Analysis Engine"
    PAGE_ICON = "🔬"
    LAYOUT = "wide"
    
    # =============================================================================
    # DEVELOPMENT & DEBUGGING
    # =============================================================================
    
    # Debug mode
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Logging level
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Test mode (smaller datasets for faster development)
    TEST_MODE = os.getenv('TEST_MODE', 'True').lower() == 'true'
    
    @classmethod
    def get_max_papers(cls):
        """Get maximum papers based on current mode"""
        if cls.TEST_MODE:
            return cls.MAX_PAPERS_TEST
        else:
            return cls.MAX_PAPERS_DEFAULT
    
    @classmethod
    def get_api_delay(cls, source: str) -> int:
        """Get API delay for specific source"""
        delays = {
            'arxiv': cls.ARXIV_DELAY,
            'pubmed': cls.PUBMED_DELAY,
            'semantic_scholar': cls.SEMANTIC_SCHOLAR_DELAY
        }
        return delays.get(source.lower(), 1)
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check required email for PubMed
        if cls.EMAIL == 'your.email@example.com':
            errors.append("Please set a valid EMAIL in .env file for PubMed API")
        
        # Check data directories exist
        import os
        if not os.path.exists(cls.DATA_DIR):
            os.makedirs(cls.DATA_DIR, exist_ok=True)
            os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
            os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
            os.makedirs(cls.SAMPLE_DATA_DIR, exist_ok=True)
        
        return errors

# Create global config instance
config = Config()

# Validate configuration on import
if __name__ == "__main__":
    errors = Config.validate_config()
    if errors:
        print("Configuration Issues:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration valid!")
        print(f"Email: {Config.EMAIL}")
        print(f"Max papers: {Config.get_max_papers()}")
        print(f"Test mode: {Config.TEST_MODE}")