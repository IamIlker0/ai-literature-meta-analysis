"""
Base Component Class for Dashboard Components
Provides consistent interface and error handling
"""

import streamlit as st
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import traceback

class ComponentBase(ABC):
    """Base class for all dashboard components"""
    
    def __init__(self, name: str, session_manager=None):
        self.name = name
        self.session_manager = session_manager
        self.error_container = None
    
    @abstractmethod
    def render(self) -> None:
        """Render the component content"""
        pass
    
    def safe_render(self) -> None:
        """Safely render component with error handling"""
        try:
            # Create error container at top of component
            self.error_container = st.container()
            
            # Render main content
            self.render()
            
        except Exception as e:
            self._handle_error(e)
    
    def _handle_error(self, error: Exception):
        """Handle component errors gracefully"""
        error_msg = f"Component '{self.name}' encountered an error: {str(error)}"
        
        with self.error_container:
            st.error(f"❌ {error_msg}")
            
            # Show details in expander for debugging
            with st.expander("🔍 Error Details (Debug)"):
                st.code(traceback.format_exc())
        
        # Log error (you can extend this to proper logging)
        print(f"COMPONENT ERROR [{self.name}]: {error}")
        print(traceback.format_exc())
    
    def show_info(self, message: str):
        """Show info message"""
        st.info(f"ℹ️ {message}")
    
    def show_success(self, message: str):
        """Show success message"""
        st.success(f"✅ {message}")
    
    def show_warning(self, message: str):
        """Show warning message"""
        st.warning(f"⚠️ {message}")
    
    def show_error(self, message: str):
        """Show error message"""
        st.error(f"❌ {message}")
    
    def create_metrics_row(self, metrics: Dict[str, Any]):
        """Create a row of metrics"""
        cols = st.columns(len(metrics))
        
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(value, dict):
                    st.metric(label, value.get('value', 0), value.get('delta'))
                else:
                    st.metric(label, value)
    
    def create_two_column_layout(self):
        """Create standard two-column layout"""
        return st.columns(2)
    
    def create_three_column_layout(self):
        """Create standard three-column layout"""
        return st.columns(3)
    
    def create_progress_tracker(self, steps: list, current_step: int):
        """Create progress tracker"""
        progress_container = st.container()
        
        with progress_container:
            # Progress bar
            progress_percent = (current_step / len(steps)) * 100
            st.progress(progress_percent / 100)
            
            # Step indicators
            cols = st.columns(len(steps))
            for i, step in enumerate(steps):
                with cols[i]:
                    if i < current_step:
                        st.success(f"✅ {step}")
                    elif i == current_step:
                        st.info(f"🔄 {step}")
                    else:
                        st.text(f"⏳ {step}")
        
        return progress_container