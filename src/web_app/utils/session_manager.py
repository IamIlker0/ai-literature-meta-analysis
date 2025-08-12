"""
Session State Management for AI Literature Analysis Dashboard
Provides centralized, stable session state handling
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional

class SessionManager:
    """Centralized session state management"""
    
    def __init__(self):
        self._init_core_state()
    
    def _init_core_state(self):
        """Initialize core session state variables"""
        
        # System components (initialized once)
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
            st.session_state.config = None
            st.session_state.collector = None
            st.session_state.analyzer = None
        
        # Navigation state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'overview'
        
        # Data state
        if 'papers' not in st.session_state:
            st.session_state.papers = []
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        
        # Collection state
        if 'collection_history' not in st.session_state:
            st.session_state.collection_history = []
        
        if 'current_query' not in st.session_state:
            st.session_state.current_query = ""
        
        # UI state
        if 'ui_settings' not in st.session_state:
            st.session_state.ui_settings = {
                'theme': 'dark',
                'show_advanced': False,
                'auto_analysis': True
            }
        
        # Analytics
        if 'session_analytics' not in st.session_state:
            st.session_state.session_analytics = {
                'session_start': datetime.now(),
                'total_papers_collected': 0,
                'total_analyses_run': 0,
                'queries_made': 0
            }
    
    def initialize_system(self):
        """Initialize system components safely"""
        if not st.session_state.system_initialized:
            try:
                # Import here to avoid circular imports
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                
                from src.config.settings import Config
                from src.data_collection.arxiv_collector import ArXivCollector
                from src.analysis.nlp_pipeline import LiteratureNLPAnalyzer
                
                with st.spinner("🔄 Initializing AI system components..."):
                    st.session_state.config = Config()
                    st.session_state.collector = ArXivCollector(st.session_state.config)
                    st.session_state.analyzer = LiteratureNLPAnalyzer(st.session_state.config)
                    st.session_state.system_initialized = True
                
                st.success("✅ System initialized successfully!")
                return True
                
            except Exception as e:
                st.error(f"❌ System initialization failed: {e}")
                return False
        
        return True
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        return {
            'papers_collected': len(st.session_state.papers),
            'analyses_run': st.session_state.session_analytics['total_analyses_run'],
            'queries_made': st.session_state.session_analytics['queries_made'],
            'session_duration': (datetime.now() - st.session_state.session_analytics['session_start']).seconds // 60,
            'system_ready': st.session_state.system_initialized
        }
    
    def update_analytics(self, event_type: str, value: int = 1):
        """Update session analytics"""
        if event_type == 'papers_collected':
            st.session_state.session_analytics['total_papers_collected'] += value
        elif event_type == 'analysis_run':
            st.session_state.session_analytics['total_analyses_run'] += value
        elif event_type == 'query_made':
            st.session_state.session_analytics['queries_made'] += value
    
    def set_current_page(self, page: str):
        """Set current page for navigation"""
        st.session_state.current_page = page
    
    def get_current_page(self) -> str:
        """Get current page"""
        return st.session_state.current_page
    
    def clear_data(self, data_type: str = 'all'):
        """Clear specific data types"""
        if data_type in ['all', 'papers']:
            st.session_state.papers = []
        
        if data_type in ['all', 'analysis']:
            st.session_state.analysis_results = None
        
        if data_type in ['all', 'history']:
            st.session_state.collection_history = []
        
        if data_type == 'all':
            st.session_state.current_query = ""