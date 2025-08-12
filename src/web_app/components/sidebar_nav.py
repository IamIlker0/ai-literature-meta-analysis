"""
Sidebar Navigation Component
Professional navigation with system status
"""

import streamlit as st
from utils.component_base import ComponentBase
from utils.session_manager import SessionManager

class SidebarNavigation(ComponentBase):
    """Professional sidebar navigation component"""
    
    def __init__(self, session_manager: SessionManager):
        super().__init__("SidebarNavigation", session_manager)
        
        # Navigation pages
        self.pages = {
            'overview': {
                'title': '📊 Dashboard Overview',
                'icon': '📊',
                'description': 'System status and quick insights'
            },
            'collection': {
                'title': '🔍 Data Collection',
                'icon': '🔍', 
                'description': 'Multi-source paper collection'
            },
            'analysis': {
                'title': '🧠 Analysis Engine',
                'icon': '🧠',
                'description': 'NLP pipeline and settings'
            },
            'visualizations': {
                'title': '📈 Visualizations',
                'icon': '📈',
                'description': 'Interactive charts and graphs'
            },
            # 'insights': {
            #     'title': '🔬 Research Insights',
            #     'icon': '🔬',
            #     'description': 'AI-generated insights'
            # },
            # 'export': {
            #     'title': '💾 Export & Reports',
            #     'icon': '💾',
            #     'description': 'Professional outputs'
            # }
        }
    
    def render(self):
        """Render sidebar navigation"""
        
        # Header section
        self._render_header()
        
        # System status
        self._render_system_status()
        
        # Navigation menu
        self._render_navigation_menu()
        
        # Settings section
        # self._render_settings()
        
        # Footer
        self._render_footer()
    
    def _render_header(self):
        """Render sidebar header"""
        st.sidebar.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: #1f77b4; margin-bottom: 0;'>🔬 AI Literature</h2>
            <p style='color: #666; margin-top: 0;'>Meta-Analysis Engine</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
    
    def _render_system_status(self):
        """Render system status section"""
        st.sidebar.markdown("### ⚡ System Status")
        
        # Get system stats
        stats = self.session_manager.get_system_stats()
        
        # Status indicators
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            status_color = "🟢" if stats['system_ready'] else "🔴"
            st.markdown(f"{status_color} **System**")
            st.markdown(f"📄 **Papers**: {stats['papers_collected']}")
        
        with col2:
            st.markdown(f"🧠 **Analyses**: {stats['analyses_run']}")
            st.markdown(f"⏱️ **Session**: {stats['session_duration']}m")
        
        # System readiness
        if not stats['system_ready']:
            if st.sidebar.button("🔄 Initialize System", type="primary"):
                self.session_manager.initialize_system()
                st.rerun()
        
        st.sidebar.markdown("---")
    
    def _render_navigation_menu(self):
        """Render main navigation menu"""
        st.sidebar.markdown("### 🧭 Navigation")
        
        current_page = self.session_manager.get_current_page()
        
        # Create navigation buttons
        for page_key, page_info in self.pages.items():
            
            # Button styling based on current page
            button_type = "primary" if page_key == current_page else "secondary"
            
            # Create button with icon and title
            if st.sidebar.button(
                f"{page_info['icon']} {page_info['title'].split(' ', 1)[1]}",  # Remove emoji from title
                key=f"nav_{page_key}",
                type=button_type,
                help=page_info['description'],
                use_container_width=True
            ):
                self.session_manager.set_current_page(page_key)
                st.rerun()
        
        st.sidebar.markdown("---")
    
    # def _render_settings(self):
    #     """Render settings section"""
    #     st.sidebar.markdown("### ⚙️ Quick Settings")
        
    #     # Theme toggle
    #     current_theme = st.session_state.ui_settings.get('theme', 'dark')
    #     theme = st.sidebar.selectbox(
    #         "🎨 Theme",
    #         options=['dark', 'light'],
    #         index=0 if current_theme == 'dark' else 1,
    #         key="theme_selector"
    #     )
    #     st.session_state.ui_settings['theme'] = theme
        
    #     # Advanced options toggle
    #     show_advanced = st.sidebar.checkbox(
    #         "🔧 Advanced Options",
    #         value=st.session_state.ui_settings.get('show_advanced', False),
    #         key="advanced_toggle"
    #     )
    #     st.session_state.ui_settings['show_advanced'] = show_advanced
        
    #     # Auto-analysis toggle
    #     auto_analysis = st.sidebar.checkbox(
    #         "⚡ Auto-run Analysis",
    #         value=st.session_state.ui_settings.get('auto_analysis', True),
    #         key="auto_analysis_toggle",
    #         help="Automatically run analysis after paper collection"
    #     )
    #     st.session_state.ui_settings['auto_analysis'] = auto_analysis
        
    #     st.sidebar.markdown("---")
    
    def _render_footer(self):
        """Render sidebar footer"""
        
        # # Quick actions
        # st.sidebar.markdown("### 🚀 Quick Actions")
        
        # # Clear data button
        # if st.sidebar.button("🗑️ Clear Data", help="Clear collected papers and analysis"):
        #     self.session_manager.clear_data('all')
        #     st.sidebar.success("Data cleared!")
        #     st.rerun()
        
        # # Refresh system button
        # if st.sidebar.button("🔄 Refresh System", help="Restart system components"):
        #     st.session_state.system_initialized = False
        #     self.session_manager.initialize_system()
        #     st.rerun()
        
        # System info
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        <div style='text-align: center; font-size: 0.8em; color: #666;'>
            <p><strong>AI Meta-Analysis v1.0</strong></p>
            <p>Powered by Sentence-BERT<br>DBSCAN • Plotly • Streamlit</p>
        </div>
        """, unsafe_allow_html=True)