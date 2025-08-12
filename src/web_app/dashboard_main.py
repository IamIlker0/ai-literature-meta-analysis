"""
AI Literature Meta-Analysis Engine - Main Dashboard Application
modular dashboard with component-based architecture
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.component_base import ComponentBase
from utils.session_manager import SessionManager

# Import components
from components.sidebar_nav import SidebarNavigation
from components.data_collection import DataCollectionComponent
from components.analysis_engine import AnalysisEngineComponent
from components.visualizations import VisualizationsComponent

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Literature Meta-Analysis Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI-powered scientific literature analysis platform"
    }
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================

def load_custom_css():
    """Load custom styling"""
    st.markdown("""
    <style>
        /* Main app styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #1f77b4, #42a5f5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .component-container {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #1f77b4;
        }
        
        .status-card {
            background: linear-gradient(135deg, #48bb78 0%, #245B80 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem;
        }
        
        .metric-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .success-gradient {
            background: linear-gradient(90deg, #56ab2f, #a8e6cf);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .warning-gradient {
            background: linear-gradient(90deg, #f093fb, #f5576c);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 20px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #1e3c72, #2a5298);
        }
        
        /* Progress bar styling */
        .stProgress > div > div {
            background: linear-gradient(90deg, #1f77b4, #42a5f5);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
        }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE COMPONENTS
# =============================================================================

class OverviewComponent(ComponentBase):
    """Dashboard overview component"""
    
    def render(self):
        st.markdown("""
            <div style="
                text-align: center; 
                margin-bottom: 2rem; 
                padding: 2rem;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.1);
            ">
                <h1 style="color: white; margin: 0 0 0.5rem 0; font-size: 2.2rem; font-weight: 700;">
                    🔬 AI Literature Meta-Analysis Engine
                </h1>
                <h3 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 600;">
                    AI-Powered Research Intelligence Platform
                </h3>
                <p style="color: #f8f9fa; margin: 0.5rem 0 0 0; font-size: 1rem;">
                    Automated collection • Advanced NLP analysis • Interactive visualizations • Research insights
                </p>
            </div>
            """, unsafe_allow_html=True)
        # System status overview
        self._render_system_overview()
        
        # Quick stats
        self._render_quick_stats()
        
        # Recent activity
        self._render_recent_activity()
    
    def _render_system_overview(self):
        """Render system status overview"""
        st.subheader("⚡ System Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="status-card">
                <h4>🤖 AI Engine</h4>
                <p>Sentence-BERT Ready</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="status-card">
                <h4>🌐 Data Sources</h4>
                <p>ArXiv • PubMed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="status-card">
                <h4>📊 Analytics</h4>
                <p>Real-time Processing</p>
            </div>
            """, unsafe_allow_html=True)
        
        # with col4:
        #     st.markdown("""
        #     <div class="status-card">
        #         <h4>🔬 Insights</h4>
        #         <p>AI-Powered</p>
        #     </div>
        #     """, unsafe_allow_html=True)
    
    def _render_quick_stats(self):
        """Render quick statistics"""
        st.subheader("📈 Session Statistics")
        
        stats = self.session_manager.get_system_stats()
        
        # Create metrics
        metrics = {
            "Papers Collected": stats['papers_collected'],
            "Analyses Run": stats['analyses_run'], 
            "Queries Made": stats['queries_made'],
            "Session Duration": f"{stats['session_duration']}min"
        }
        
        self.create_metrics_row(metrics)
    
    def _render_recent_activity(self):
        """Render recent activity"""
        st.subheader("🕐 Recent Activity")
        
        if st.session_state.collection_history:
            # Show recent queries
            recent_queries = st.session_state.collection_history[-3:]  # Last 3
            
            for i, query_data in enumerate(reversed(recent_queries), 1):
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Query #{len(st.session_state.collection_history) - i + 1}</strong><br>
                    <em>"{query_data.get('query', 'Unknown')}"</em><br>
                    <small>{query_data.get('papers_count', 0)} papers • {query_data.get('timestamp', 'Unknown time')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("🔍 No recent activity. Start by collecting some papers!")
            
            # Quick start buttons
            st.markdown("### 🚀 Quick Start")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🤖 AI & Machine Learning", type="primary"):
                    self.session_manager.set_current_page('collection')
                    st.session_state.quick_query = "artificial intelligence machine learning"
                    st.rerun()
            
            with col2:
                if st.button("🧬 Computational Biology"):
                    self.session_manager.set_current_page('collection')
                    st.session_state.quick_query = "computational biology bioinformatics"
                    st.rerun()
            
            with col3:
                if st.button("🌐 Computer Vision"):
                    self.session_manager.set_current_page('collection')
                    st.session_state.quick_query = "computer vision deep learning"
                    st.rerun()


class PlaceholderComponent(ComponentBase):
    """Placeholder component for pages under development"""
    
    def __init__(self, page_name: str, session_manager):
        super().__init__(f"Placeholder_{page_name}", session_manager)
        self.page_name = page_name
    
    def render(self):
        page_info = {
            'collection': {
                'title': '🔍 Data Collection Engine',
                'description': 'Multi-source academic paper collection with real-time progress tracking',
                'features': [
                    '🌐 ArXiv, PubMed, Semantic Scholar integration',
                    '📋 Smart query builder with suggestions',
                    '⚡ Real-time collection progress',
                    '🔍 Advanced filtering options',
                    '📊 Collection analytics and history'
                ]
            },
            'analysis': {
                'title': '🧠 Analysis Engine',
                'description': 'Advanced NLP pipeline with customizable parameters',
                'features': [
                    '🤖 Sentence-BERT semantic analysis',
                    '🔗 DBSCAN clustering with parameter tuning',
                    '👥 Author network analysis',
                    '📈 Research trend detection',
                    '🎯 Custom analysis modes'
                ]
            },
            'visualizations': {
                'title': '📈 Interactive Visualizations',
                'description': 'Rich, interactive charts and knowledge graphs',
                'features': [
                    '🕸️ Interactive knowledge graphs',
                    '📊 Dynamic clustering visualizations',
                    '🗺️ Research landscape mapping',
                    '📈 Temporal trend analysis',
                    '🎨 Customizable chart themes'
                ]
            },
            # # 'insights': {
            # #     'title': '🔬 Research Insights',
            # #     'description': 'AI-powered research gap analysis and recommendations',
            # #     'features': [
            # #         '🤖 Automated research gap detection',
            # #         '🎯 Smart paper recommendations',
            # #         '📋 Executive summary generation',
            # #         '🔮 Trend predictions',
            # #         '🤝 Collaboration opportunity identification'
            # #     ]
            # # },
        }
        
        info = page_info.get(self.page_name, {})
        
        # # Header
        # st.markdown(f'<h1 class="main-header">{info.get("title", "🚧 Under Development")}</h1>', 
        #            unsafe_allow_html=True)
        
        # # Description
        # if info.get('description'):
        #     st.markdown(f"""
        #     <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        #         <h3>{info['description']}</h3>
        #     </div>
        #     """, unsafe_allow_html=True)
        
        # # Development status
        # st.markdown("""
        # <div class="warning-gradient">
        #     <h3>🚧 Component Under Development</h3>
        #     <p>This component is being developed using modular architecture. Check back soon!</p>
        # </div>
        # """, unsafe_allow_html=True)
        
        # # Planned features
        # if info.get('features'):
        #     st.subheader("🎯 Planned Features")
            
        #     col1, col2 = st.columns(2)
            
        #     for i, feature in enumerate(info['features']):
        #         with col1 if i % 2 == 0 else col2:
        #             st.markdown(f"- {feature}")
        
        # # Development timeline
        # st.subheader("📅 Development Timeline")
        
        # timeline_data = {
        #     'collection': {'status': '🔄 In Progress', 'eta': 'Next 1-2 days'},
        #     'analysis': {'status': '📋 Planned', 'eta': 'Day 3-4'},
        #     'visualizations': {'status': '📋 Planned', 'eta': 'Day 4-5'},
        #     'insights': {'status': '📋 Planned', 'eta': 'Day 5-6'},
        #     'export': {'status': '📋 Planned', 'eta': 'Day 6-7'}
        # }
        
        # current_status = timeline_data.get(self.page_name, {})
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     st.metric("Development Status", current_status.get('status', 'Unknown'))
        
        # with col2:
        #     st.metric("Estimated Completion", current_status.get('eta', 'TBD'))


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    
    # Load custom styling
    load_custom_css()
    
    # Initialize session manager
    session_manager = SessionManager()
    
    # Initialize system if needed
    if not st.session_state.system_initialized:
        session_manager.initialize_system()
    
    # Render sidebar navigation
    sidebar_nav = SidebarNavigation(session_manager)
    sidebar_nav.safe_render()
    
    # Get current page
    current_page = session_manager.get_current_page()
    
    # Render main content based on current page
    # Render main content based on current page
    if current_page == 'overview':
        overview_component = OverviewComponent("Overview", session_manager)
        overview_component.safe_render()

    elif current_page == 'collection':
        collection_component = DataCollectionComponent(session_manager)
        collection_component.safe_render()

    elif current_page == 'analysis':
        analysis_component = AnalysisEngineComponent(session_manager)
        analysis_component.safe_render()

    elif current_page == 'visualizations':
        viz_component = VisualizationsComponent(session_manager)
        viz_component.safe_render()
        
    else:
        # Render placeholder for other components
        placeholder_component = PlaceholderComponent(current_page, session_manager)
        placeholder_component.safe_render()
            
    # Footer
    render_footer()


def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><b>AI-Powered Scientific Literature Meta-Analysis Engine v1.0</b></p>
        <p>🏗️ Modular Architecture • 🤖 Sentence-BERT • 📊 Interactive Visualizations • 🔬 Research Intelligence</p>
        <p><i>Built with Streamlit • Component-based Design • Development Practices</i></p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()