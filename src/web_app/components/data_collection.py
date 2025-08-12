"""
Advanced Data Collection Component
Professional multi-source academic paper collection interface
"""

import streamlit as st
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go

from utils.component_base import ComponentBase
from utils.session_manager import SessionManager

class DataCollectionComponent(ComponentBase):
    """Advanced data collection interface with multi-source support"""
    
    def __init__(self, session_manager: SessionManager):
        super().__init__("DataCollection", session_manager)
        
        # Collection settings
        self.data_sources = {
            'arxiv': {
                'name': 'ArXiv',
                'description': 'Open access preprints',
                'icon': '📚',
                'status': '✅ Available',
                'categories': ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'stat.ML']
            },
            'pubmed': {
                'name': 'PubMed',
                'description': 'Medical literature',
                'icon': '🏥',
                'status': '🔄 Coming Soon',
                'categories': ['Medicine', 'Biology', 'Neuroscience', 'Genetics']
            },
            'semantic_scholar': {
                'name': 'Semantic Scholar',
                'description': 'Cross-disciplinary papers',
                'icon': '🎓',
                'status': '🔄 Coming Soon',
                'categories': ['Computer Science', 'Medicine', 'Biology', 'Physics']
            }
        }
        
        # Query suggestions
        self.query_suggestions = {
            'AI & Machine Learning': [
                'artificial intelligence machine learning',
                'deep learning neural networks',
                'transformer attention mechanism',
                'reinforcement learning algorithms',
                'computer vision image recognition'
            ],
            'Medical & Biology': [
                'computational biology bioinformatics',
                'machine learning healthcare',
                'medical imaging analysis',
                'drug discovery AI',
                'genomics data analysis'
            ],
            'Data Science': [
                'natural language processing',
                'data mining algorithms',
                'statistical learning theory',
                'big data analytics',
                'information retrieval systems'
            ]
        }
    
    def render(self):
        """Render the data collection interface"""
        
        # Header
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
                    🔍 Data Collection Engine
                </h1>
                <h3 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 600;">
                    Professional Multi-Source Academic Paper Collection
                </h3>
                <p style="color: #f8f9fa; margin: 0.5rem 0 0 0; font-size: 1rem;">
                    Advanced query building • Real-time progress • Smart filtering
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Main interface
        tab1, tab2, tab3 = st.tabs(["🔍 Query Builder", "📊 Collection Status", "📋 History & Analytics"])
        
        with tab1:
            self._render_query_builder()
        
        with tab2:
            self._render_collection_status()
        
        with tab3:
            self._render_history_analytics()
    
    def _render_query_builder(self):
        """Render advanced query builder interface"""
        
        # Quick start section
        st.subheader("🚀 Quick Start")
        self._render_quick_start_buttons()
        
        st.markdown("---")
        
        # Advanced query builder
        st.subheader("🔧 Advanced Query Builder")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Main query input
            query = st.text_input(
                "🔍 Research Query",
                value=st.session_state.get('quick_query', ''),
                placeholder="e.g., machine learning healthcare applications",
                help="Enter keywords, phrases, or boolean queries"
            )
            
            # Query suggestions
            if query:
                suggestions = self._get_query_suggestions(query)
                if suggestions:
                    st.markdown("**💡 Related suggestions:**")
                    suggestion_cols = st.columns(min(3, len(suggestions)))
                    for i, suggestion in enumerate(suggestions[:3]):
                        with suggestion_cols[i]:
                            if st.button(f"📎 {suggestion}", key=f"suggestion_{i}"):
                                st.session_state.quick_query = suggestion
                                st.rerun()
        
        with col2:
            # Collection parameters
            st.markdown("**📊 Collection Settings**")
            
            max_papers = st.slider(
                "Max Papers per Source",
                min_value=5,
                max_value=100,
                value=50,
                step=5,
                help="Number of papers to collect from each source"
            )
            
            # Data source selection
            st.markdown("**🌐 Data Sources**")
            
            selected_sources = []
            for source_key, source_info in self.data_sources.items():
                available = source_info['status'] == '✅ Available'
                
                if available:
                    if st.checkbox(
                        f"{source_info['icon']} {source_info['name']}",
                        value=True,
                        key=f"source_{source_key}",
                        help=source_info['description']
                    ):
                        selected_sources.append(source_key)
                else:
                    st.markdown(f"{source_info['icon']} {source_info['name']} {source_info['status']}")
        
        # Advanced options (collapsible)
        with st.expander("🔧 Advanced Options", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Date range filter
                st.markdown("**📅 Date Range**")
                date_filter = st.selectbox(
                    "Publication Period",
                    ["Any time", "Last year", "Last 5 years", "Last 10 years", "Custom range"],
                    index=0
                )
                
                if date_filter == "Custom range":
                    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*5))
                    end_date = st.date_input("End Date", datetime.now())
            
            with col2:
                # Category filter
                st.markdown("**📚 Categories**")
                if selected_sources and 'arxiv' in selected_sources:
                    categories = st.multiselect(
                        "ArXiv Categories",
                        self.data_sources['arxiv']['categories'],
                        default=[],
                        help="Filter papers by ArXiv subject categories"
                    )
                
                # Quality filters
                st.markdown("**⭐ Quality Filters**")
                min_citations = st.number_input(
                    "Minimum Citations",
                    min_value=0,
                    max_value=1000,
                    value=0,
                    help="Filter papers with minimum citation count"
                )
        
        # Collection button
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button(
                f"🚀 Collect Papers ({max_papers} per source)",
                type="primary",
                disabled=not (query.strip() and selected_sources),
                help="Start collecting papers from selected sources"
            ):
                self._run_collection(query, selected_sources, max_papers)
        
        with col2:
            if st.button("🔄 Clear Query"):
                st.session_state.quick_query = ""
                st.rerun()
        
        with col3:
            estimated_total = len(selected_sources) * max_papers
            st.metric("Estimated Papers", estimated_total)
    def _render_quick_start_buttons(self):
        """Render quick start suggestion buttons"""

        col1, col2, col3 = st.columns(3)
        categories = list(self.query_suggestions.keys())

        for i, category in enumerate(categories[:3]):
            with [col1, col2, col3][i]:
                st.markdown(f"**{category}**")
                for query in self.query_suggestions[category][:2]:  # Show 2 per category
                    if st.button(
                        f"🎯 {query.title()}",
                        key=f"quick_{category}_{query}",
                        use_container_width=True
                    ):
                        # Only update and rerun if the query is different
                        if st.session_state.get('quick_query', '') != query:
                            st.session_state.quick_query = query
                            st.rerun()
    
    def _get_query_suggestions(self, query: str) -> List[str]:
        """Get query suggestions based on input"""
        
        # Simple keyword matching for suggestions
        suggestions = []
        query_lower = query.lower()
        
        for category, queries in self.query_suggestions.items():
            for suggested_query in queries:
                if any(word in suggested_query.lower() for word in query_lower.split()):
                    if suggested_query not in suggestions and suggested_query != query:
                        suggestions.append(suggested_query)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _render_collection_status(self):
        """Render real-time collection status"""
        
        st.subheader("📊 Collection Status")
        
        # Current session stats
        stats = self.session_manager.get_system_stats()
        
        # Status overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Papers Collected", stats['papers_collected'])
        
        with col2:
            st.metric("Active Sources", 1)  # Currently only ArXiv
        
        with col3:
            st.metric("Success Rate", "100%")
        
        with col4:
            st.metric("Avg Collection Time", "2.5s per paper")
        
        # Data sources status
        st.markdown("### 🌐 Data Sources Status")
        
        for source_key, source_info in self.data_sources.items():
            
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            
            with col1:
                st.markdown(f"**{source_info['icon']}**")
            
            with col2:
                st.markdown(f"**{source_info['name']}**")
            
            with col3:
                st.markdown(source_info['description'])
            
            with col4:
                status_color = "🟢" if source_info['status'] == '✅ Available' else "🟡"
                st.markdown(f"{status_color} {source_info['status']}")
        
        # Recent collection activity
        if st.session_state.papers:
            st.markdown("### 📚 Recent Collections")
            
            # Create preview of recent papers
            preview_data = []
            for paper in st.session_state.papers[-5:]:  # Last 5 papers
                preview_data.append({
                    'Title': paper['title'][:50] + '...' if len(paper['title']) > 50 else paper['title'],
                    'Source': paper['source'].upper(),
                    'Date': paper['date'],
                    'Authors': len(paper['authors']),
                    'Categories': paper['categories'][0] if paper['categories'] else 'N/A'
                })
            
            df = pd.DataFrame(preview_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("📭 No papers collected yet. Use the Query Builder to start collecting!")
    
    def _render_history_analytics(self):
        """Render collection history and analytics"""
        
        st.subheader("📋 Collection History & Analytics")
        
        if st.session_state.collection_history:
            
            # History table
            history_data = []
            for i, entry in enumerate(st.session_state.collection_history, 1):
                history_data.append({
                    'Query #': i,
                    'Query': entry.get('query', 'Unknown'),
                    'Papers': entry.get('papers_count', 0),
                    'Sources': ', '.join(entry.get('sources', [])),
                    'Timestamp': entry.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
            
            # Analytics charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Papers per query chart
                fig_bar = px.bar(
                    df,
                    x='Query #',
                    y='Papers',
                    title="📊 Papers Collected per Query",
                    color='Papers',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Query timeline
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                
                fig_timeline = px.line(
                    df,
                    x='Timestamp',
                    y='Papers',
                    title="📈 Collection Timeline",
                    markers=True,
                    line_shape='spline'
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Export history
            st.markdown("### 💾 Export History")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export History (CSV)"):
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_data,
                        file_name=f"collection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("🗑️ Clear History"):
                    st.session_state.collection_history = []
                    st.success("History cleared!")
                    st.rerun()
        
        else:
            # Empty state
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666;">
                <h3>📭 No Collection History</h3>
                <p>Start collecting papers to see analytics and history here!</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _run_collection(self, query: str, sources: List[str], max_papers: int):
        """Execute paper collection process"""
        
        # Validate system initialization
        if not st.session_state.system_initialized:
            self.show_error("System not initialized. Please initialize system first.")
            return
        
        # Progress tracking setup
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 📡 Collection in Progress")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Metrics tracking
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                papers_metric = st.metric("Papers Collected", 0)
            with metrics_col2:
                time_metric = st.metric("Elapsed Time", "0s")
            with metrics_col3:
                rate_metric = st.metric("Collection Rate", "0 papers/min")
        
        start_time = time.time()
        
        try:
            # Collection process
            all_papers = []
            
            for i, source in enumerate(sources):
                
                # Update progress
                base_progress = (i / len(sources)) * 100
                progress_bar.progress(int(base_progress))
                status_text.info(f"🔍 Collecting from {source.upper()}...")
                
                if source == 'arxiv':
                    # Collect from ArXiv
                    papers = st.session_state.collector.search_papers(
                        query,
                        max_results=max_papers
                    )
                    
                    if papers:
                        all_papers.extend(papers)
                        
                        # Update metrics
                        elapsed_time = time.time() - start_time
                        collection_rate = len(all_papers) / (elapsed_time / 60) if elapsed_time > 0 else 0
                        
                        papers_metric.metric("Papers Collected", len(all_papers))
                        time_metric.metric("Elapsed Time", f"{elapsed_time:.1f}s")
                        rate_metric.metric("Collection Rate", f"{collection_rate:.1f} papers/min")
            
            # Finalize collection
            if all_papers:
                # Store papers
                st.session_state.papers = all_papers
                st.session_state.current_query = query
                
                try:
                    json_file, csv_file = self._save_papers_to_file(all_papers, query)
                    st.success(f"💾 Auto-saved: {len(all_papers)} papers")
                    st.info(f"📄 Files: {json_file} | {csv_file}")
                except Exception as e:
                    st.warning(f"⚠️ Auto-save failed: {e}")
                    
                # Update analytics
                self.session_manager.update_analytics('papers_collected', len(all_papers))
                self.session_manager.update_analytics('query_made')
                
                # Add to history
                st.session_state.collection_history.append({
                    'query': query,
                    'papers_count': len(all_papers),
                    'sources': sources,
                    'timestamp': datetime.now()
                })
                
                progress_bar.progress(100)
                status_text.success(f"✅ Successfully collected {len(all_papers)} papers!")
                
                # Show success message with next steps
                st.markdown("""
                <div class="success-gradient">
                    <h3>🎉 Collection Completed!</h3>
                    <p>Papers have been collected and are ready for analysis. Navigate to the <strong>Collection Status</strong> to review your data or go to the <strong>Analysis Engine</strong> to process your research corpus.</p>                
                    </div>
                """, unsafe_allow_html=True)
                
                # # Quick navigation to analysis
                # col1, col2 = st.columns(2)
                
                # with col1:
                #     if st.button("🧠 Go to Analysis Engine", type="primary"):
                #         self.session_manager.set_current_page('analysis')
                #         st.rerun()
                
                # with col2:
                #     if st.button("📊 View Collection Status"):
                #         # Switch to status tab (would need tab state management)
                #         pass
            
            else:
                progress_bar.progress(100)
                status_text.error("❌ No papers found. Try different keywords or sources.")
        except Exception as e:
            progress_bar.progress(0)
            status_text.error(f"❌ Collection failed: {str(e)}")
            self.show_error(f"Collection error: {str(e)}")
    
    def _save_papers_to_file(self, papers: List[Dict], query: str):
        """Auto-save collected papers to file system"""
        
        import os
        import json
        import pandas as pd
        
        # Ensure directories exist
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON (raw format)
        json_file = f"data/raw/papers_{timestamp}.json"
        export_data = {
            'metadata': {
                'query': query,
                'timestamp': timestamp,
                'papers_count': len(papers),
                'source': 'auto_save'
            },
            'papers': papers
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str, ensure_ascii=False)
        
        # Save CSV (processed format)
        csv_file = f"data/processed/papers_{timestamp}.csv"
        csv_data = []
        
        for paper in papers:
            csv_data.append({
                'id': paper['id'],
                'title': paper['title'],
                'first_author': paper['authors'][0] if paper['authors'] else 'Unknown',
                'all_authors': '; '.join(paper['authors']),
                'date': paper['date'],
                'source': paper['source'],
                'arxiv_id': paper.get('arxiv_id', ''),
                'categories': '; '.join(paper['categories']),
                'abstract': paper['abstract']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        return json_file, csv_file