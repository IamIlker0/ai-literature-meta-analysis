"""
Simplified Analysis Engine Component
Core NLP analysis functionality with clean interface
"""

import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import os
import numpy as np
from collections import Counter
import glob

from utils.component_base import ComponentBase
from utils.session_manager import SessionManager



class AnalysisEngineComponent(ComponentBase):
    """Simplified NLP analysis engine"""
    
    def __init__(self, session_manager: SessionManager):
        super().__init__("AnalysisEngine", session_manager)
    
    def render(self):
        """Render the analysis engine interface"""
        
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
                    🧠 Analysis Engine
                </h1>
                <h3 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 600;">
                    NLP Analysis Pipeline
                </h3>
                <p style="color: #f8f9fa; margin: 0.5rem 0 0 0; font-size: 1rem;">
                    Semantic clustering • Author analysis • Keyword extraction
                </p>
            </div>
            """, unsafe_allow_html=True)
        # Check if data available
        if not st.session_state.papers:
            self._render_no_data_state()
            return
        
        # Simple tabs structure
        tab1, tab2, tab3 = st.tabs([
            "⚙️ Configuration", 
            "📊 Results", 
            "💾 Export"
        ])
        
        with tab1:
            self._render_configuration()
        
        with tab2:
            self._render_results()
        
        with tab3:
            self._render_export()
    
    def _render_no_data_state(self):
        """Render state when no data is available"""
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 50%, #fdcb6e 100%);
            border: 1px solid #f39c12;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(243, 156, 18, 0.2);
        ">
            <h3 style="color: #8b5a00; margin-bottom: 1rem;">⚠️ No Data Available</h3>
            <p style="color: #7d6608; margin-bottom: 0;">You can either collect new papers or load from saved files.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Two options: Collect new OR Load from files
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Collect New Papers")
            st.markdown("Collect fresh papers from online sources")
            if st.button("🔍 Go to Data Collection", type="primary", use_container_width=True):
                self.session_manager.set_current_page('collection')
                st.rerun()
        
        with col2:
            st.markdown("### 📂 Load Saved Data")
            st.markdown("Use previously collected papers")
            self._render_file_selector()
            
    def _render_file_selector(self):
        """Render file selection interface"""
        
        # Find saved files
        raw_files = glob.glob("data/raw/papers_*.json")
        processed_files = glob.glob("data/processed/papers_*.csv") 
        
        all_files = raw_files + processed_files
        
        if not all_files:
            st.info("📁 No saved files found")
            return
        
        # Format file list for display
        file_options = {}
        for file_path in all_files:
            # Extract timestamp from filename
            filename = os.path.basename(file_path)
            if filename.startswith('papers_'):
                timestamp_part = filename.replace('papers_', '').replace('.json', '').replace('.csv', '')
                display_name = f"📄 {timestamp_part} ({os.path.splitext(filename)[1]})"
                file_options[display_name] = file_path
        
        if file_options:
            selected_display = st.selectbox(
                "Select saved data file:",
                options=list(file_options.keys()),
                index=0
            )
            
            if st.button("📥 Load Selected File", use_container_width=True):
                selected_file = file_options[selected_display]
                self._load_data_from_file_safe(selected_file)
                
    def _load_data_from_file_safe(self, file_path):
        """Load papers from selected file - SAFE VERSION"""
        
        try:
            import json
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract papers
            if 'papers' in data:
                papers = data['papers']
            elif isinstance(data, list):
                papers = data
            else:
                papers = [data]
            
            # Save to session state
            st.session_state.papers = papers
            
            # SUCCESS MESSAGE - NO RERUN!
            st.success(f"✅ Loaded {len(papers)} papers from {os.path.basename(file_path)}")
            
            # DON'T CALL st.rerun() HERE!
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    def _render_configuration(self):
        """Simplified configuration interface"""
        
        papers_count = len(st.session_state.papers)
        st.success(f"📊 Ready to analyze **{papers_count} papers**")
        
        # ADD FILE SELECTOR HERE - ALWAYS VISIBLE
        st.markdown("---")
        st.markdown("### 📂 Data Management")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Current Dataset:**")
            st.info(f"📄 {papers_count} papers loaded")
        
        with col2:
            st.markdown("**Load Different Dataset:**")
            self._render_file_selector_compact()
        
        st.markdown("---")
        # Simple configuration
        col1, col2 = st.columns(2)
        
        with col1:
            # Analysis mode
            mode = st.selectbox(
                "Analysis Mode:",
                options=['quick', 'standard', 'deep'],
                format_func=lambda x: {
                    'quick': '⚡ Quick (30s)',
                    'standard': '🔄 Standard (1-2min)', 
                    'deep': '🔬 Deep (2-3min)'
                }[x],
                index=1
            )
            st.session_state.analysis_mode = mode
        
        with col2:
            # Clustering method
            algorithm = st.selectbox(
                "Clustering Method:",
                options=['dbscan', 'kmeans'],
                format_func=lambda x: {
                    'dbscan': 'DBSCAN (Automatic)',
                    'kmeans': 'K-Means (Manual)'
                }[x],
                index=0
            )
            st.session_state.clustering_algorithm = algorithm
        
        # Advanced options (collapsed by default)
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    0.5, 0.9, 0.7, 0.05
                )
                st.session_state.similarity_threshold = similarity_threshold
            
            with col2:
                if algorithm == 'kmeans':
                    n_clusters = st.slider(
                        "Number of Clusters",
                        2, 10, 4
                    )
                    st.session_state.kmeans_n_clusters = n_clusters
        
        # Run analysis button
        st.markdown("---")
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            self._run_analysis()
            
    def _render_file_selector_compact(self):
        """Compact file selection for configuration tab"""        
        # Find saved files
        raw_files = glob.glob("data/raw/papers_*.json")
        
        if not raw_files:
            st.info("No saved files")
            return
        
        # Format file list
        file_options = []
        file_paths = {}
        
        file_options.append("Current Dataset")
        file_paths["Current Dataset"] = None
        
        for file_path in raw_files:
            filename = os.path.basename(file_path)
            if filename.startswith('papers_'):
                timestamp_part = filename.replace('papers_', '').replace('.json', '')
                display_name = f"{timestamp_part}"
                file_options.append(display_name)
                file_paths[display_name] = file_path
        
        # SAFE: Use session state to track selection
        if 'selected_dataset' not in st.session_state:
            st.session_state.selected_dataset = "Current Dataset"
        
        selected_display = st.selectbox(
            "Switch dataset:",
            options=file_options,
            index=file_options.index(st.session_state.selected_dataset),
            key="dataset_selector"
        )
        
        # SAFE: Only load if different from current AND button clicked
        if selected_display != st.session_state.selected_dataset:
            if st.button("🔄 Switch Dataset", key="switch_dataset_btn"):
                if selected_display != "Current Dataset":
                    selected_file = file_paths[selected_display]
                    self._load_data_from_file_safe(selected_file)
                    st.session_state.selected_dataset = selected_display
                else:
                    st.session_state.selected_dataset = "Current Dataset"
                    
                    
    def _run_analysis(self):
        """Simplified analysis pipeline"""
        
        if not st.session_state.system_initialized:
            st.error("System not initialized. Please initialize first.")
            return
        
        papers = st.session_state.papers
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        try:
            # Step 1: Initialize
            progress_bar.progress(20)
            status_text.info("🔄 Initializing analysis...")
            time.sleep(0.5)
            
            # Step 2: Run analysis
            progress_bar.progress(50)
            status_text.info("🧠 Running NLP analysis...")
            
            # Call the actual analyzer
            results = st.session_state.analyzer.comprehensive_analysis(papers)
            
            # Step 3: Process results
            progress_bar.progress(80)
            status_text.info("📊 Processing results...")
            time.sleep(0.5)
            
            # Clean and save results
            cleaned_results = self._clean_results_for_json(results)
            
            # Save to session state
            st.session_state.analysis_results = cleaned_results
            
            # Auto-save
            self._save_results(cleaned_results)
            
            # Complete
            progress_bar.progress(100)
            elapsed_time = round(time.time() - start_time, 1)
            status_text.success(f"✅ Analysis completed in {elapsed_time}s!")
            
            # Update analytics
            self.session_manager.update_analytics('analysis_run')
            
        except Exception as e:
            progress_bar.progress(0)
            status_text.error(f"❌ Analysis failed: {str(e)}")
            st.error(f"Error details: {str(e)}")
    
    def _clean_results_for_json(self, results):
        """Clean results for JSON serialization"""
        
        def clean_value(obj):
            """Recursively clean values for JSON"""
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): clean_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_value(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(clean_value(item) for item in obj)
            else:
                return obj
        
        return clean_value(results)
    
    def _render_results(self):
        """Simplified results display"""
        
        if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
            st.info("🔄 No results available. Run analysis first.")
            return
        
        results = st.session_state.analysis_results
        
        # CRITICAL FIX: Check if results is None
        if results is None:
            st.warning("⚠️ Analysis results are None. Please run analysis again.")
            st.info("Click 'Run Analysis' in the Configuration tab to generate results.")
            return

        # ADDITIONAL FIX: Check if results is empty dict
        if not results or not isinstance(results, dict):
            st.warning("⚠️ Analysis results are empty or invalid. Please run analysis again.")
            st.info("Click 'Run Analysis' in the Configuration tab to generate results.")
            return
        
        # Summary metrics - MORE PROMINENT
        st.markdown("### 📊 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            clusters = results.get('clustering', {}).get('clusters', {})
            # Count all clusters including uncategorized
            total_clusters = len(clusters)
            distinct_clusters = len([c for c in clusters.keys() if c != '-1'])
            st.metric("Clusters Found", f"{distinct_clusters} (+1 uncategorized)")
        
        with col2:
            authors = results.get('authors', {})
            st.metric("Unique Authors", authors.get('total_authors', 0))
        
        with col3:
            keywords = results.get('global_keywords', [])
            st.metric("Keywords Extracted", len(keywords))
        
        with col4:
            papers_count = len(st.session_state.papers)
            st.metric("Papers Analyzed", papers_count)
        
        # Main results sections
        st.markdown("---")
        
        # Clustering results
        self._render_clustering_simple(results)
        
        # Keywords
        self._render_keywords_simple(results)
        
        # Authors
        self._render_authors_simple(results)
    
    def _render_clustering_simple(self, results):
        """Simple clustering visualization"""
        
        st.subheader("🔗 Clustering Results")
        
        clusters = results.get('clustering', {}).get('clusters', {})
        
        if not clusters:
            st.warning("No clustering results available.")
            return
        
        # Prepare data for visualization - INCLUDING uncategorized
        cluster_data = []
        for cluster_id, cluster_info in clusters.items():
            cluster_data.append({
                'Cluster': cluster_info.get('name', 'Uncategorized' if cluster_id == '-1' else f'Cluster {cluster_id}'),
                'Papers': cluster_info.get('size', 0)
            })
        
        if cluster_data:
            df = pd.DataFrame(cluster_data)
            
            # Show table first
            st.dataframe(df, use_container_width=True)
            
            # Get data from dataframe for plot
            cluster_names = df['Cluster'].tolist()
            paper_counts = df['Papers'].tolist()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=cluster_names,
                    y=paper_counts,
                    marker=dict(
                        color=paper_counts,
                        colorscale='Blues',
                        showscale=True if len(cluster_names) > 1 else False
                    )
                )
            ])
            fig.update_layout(
                title='Papers per Cluster',
                xaxis_title='Clusters',
                yaxis_title='Number of Papers',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No distinct clusters found. Papers are too diverse or similar.")


    def _render_keywords_simple(self, results):
        """Simple keywords display"""
        
        st.subheader("🔑 Top Keywords")
        
        keywords = results.get('global_keywords', [])
        
        if keywords:
            # Top 10 keywords - already sorted by frequency
            top_keywords = keywords[:10]
            keyword_names = results.get('keywords', [])  # ['network', 'neural', 'deep', ...]

           # Convert to proper format for DataFrame
            df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
            
          # Also show table FIRST
            st.dataframe(df, use_container_width=True)

            # SHEET'TEN VERİYİ AL VE PLOT YAP
            plot_data = df.copy()
            keywords_list = plot_data['Keyword'].tolist()
            frequencies_list = plot_data['Frequency'].tolist()

            fig = go.Figure(data=[
                go.Bar(
                    x=frequencies_list,
                    y=keywords_list,
                    orientation='h',
                    marker=dict(
                        color=frequencies_list,
                        colorscale='Blues',      # Mavi tonları
                        # colorscale='Plasma',   # Mor-sarı
                        # colorscale='Viridis',  # Yeşil-mavi-mor
                        showscale=True,
                        colorbar=dict(title="Frequency")
                    )
                )
            ])
            fig.update_layout(
                title='Top Research Keywords',
                xaxis_title='Frequency',
                yaxis_title='Keywords',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            
    def _render_authors_simple(self, results):
        """Simple author analysis"""
        
        st.subheader("👥 Author Analysis")
        
        authors = results.get('authors', {})
        most_productive = authors.get('most_productive', [])
        
        if most_productive:
            # Top 5 authors
            author_data = []
            for item in most_productive[:5]:
                # Handle both tuple and dict formats
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    author = item[0]
                    stats = item[1]
                else:
                    # If it's already in the right format
                    author = item.get('author', 'Unknown')
                    stats = item
                
                author_data.append({
                    'Author': author,
                    'Papers': stats.get('paper_count', 0),
                    'Collaborators': stats.get('collaboration_count', 0)
                })
            
            if author_data:
                df = pd.DataFrame(author_data)
                st.dataframe(df, use_container_width=True)
                
                # Summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Authors", authors.get('total_authors', 0))
                with col2:
                    st.metric("Total Collaborations", authors.get('total_collaborations', 0))
        else:
            st.info("No author data available.")
    
    def _render_export(self):
        """Simple export interface"""
        
        st.subheader("💾 Export Results")
        
        if 'analysis_results' not in st.session_state or st.session_state.analysis_results is None:
            st.info("No results to export. Run analysis first.")
            return
        
        st.markdown("### 📄 Export as JSON")
        if st.button("Generate Complete JSON Report", use_container_width=True):
            self._export_json()
        
        st.markdown("---")
        
        st.markdown("### 📊 Export as CSV Files")
        self._export_csv()
    
    def _save_results(self, results):
        """Auto-save results with proper JSON serialization"""
        
        try:
            # Create directory if needed
            os.makedirs("data/processed", exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"data/processed/analysis_{timestamp}.json"
            
            # Simple save data
            save_data = {
                'timestamp': timestamp,
                'papers_count': len(st.session_state.papers),
                'results': results
            }
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            st.success(f"💾 Results auto-saved to {filepath}")
            
        except Exception as e:
            st.warning(f"Auto-save failed: {str(e)}")
    
    def _export_json(self):
        """Export results as JSON"""
        
        results = st.session_state.analysis_results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name=f"analysis_{timestamp}.json",
            mime="application/json"
        )
    
    def _export_csv(self):
        """Export results as CSV"""
        
        results = st.session_state.analysis_results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Check if results is valid
        if results is None or not isinstance(results, dict):
            st.warning("No valid analysis results to export. Please run analysis first.")
            return

        # Create multiple CSV options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Keywords CSV
            keywords = results.get('global_keywords', [])
            if keywords:
                df_keywords = pd.DataFrame(keywords[:20], columns=['Keyword', 'Frequency'])
                csv_keywords = df_keywords.to_csv(index=False)
                
                st.download_button(
                    label="📊 Keywords CSV",
                    data=csv_keywords,
                    file_name=f"keywords_{timestamp}.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Authors CSV
            authors = results.get('authors', {})
            most_productive = authors.get('most_productive', [])
            if most_productive:
                author_data = []
                for item in most_productive[:10]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        author = item[0]
                        stats = item[1]
                    else:
                        author = item.get('author', 'Unknown')
                        stats = item
                    
                    author_data.append({
                        'Author': author,
                        'Papers': stats.get('paper_count', 0),
                        'Collaborators': stats.get('collaboration_count', 0)
                    })
                
                df_authors = pd.DataFrame(author_data)
                csv_authors = df_authors.to_csv(index=False)
                
                st.download_button(
                    label="👥 Authors CSV",
                    data=csv_authors,
                    file_name=f"authors_{timestamp}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Clusters CSV
            clusters = results.get('clustering', {}).get('clusters', {})
            if clusters:
                cluster_data = []
                for cluster_id, cluster_info in clusters.items():
                    cluster_data.append({
                        'Cluster_ID': cluster_id,
                        'Cluster_Name': cluster_info.get('name', 'Unknown'),
                        'Papers_Count': cluster_info.get('size', 0)
                    })
                
                df_clusters = pd.DataFrame(cluster_data)
                csv_clusters = df_clusters.to_csv(index=False)
                
                st.download_button(
                    label="🔗 Clusters CSV",
                    data=csv_clusters,
                    file_name=f"clusters_{timestamp}.csv",
                    mime="text/csv"
                )