# src/web_app/components/visualizations.py
"""
Simplified Visualizations Component
Clean, focused interactive charts and network graphs
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter, defaultdict
import math
import glob   
import os
import json

from utils.component_base import ComponentBase
from utils.session_manager import SessionManager

class VisualizationsComponent(ComponentBase):
    """Simplified visualization component focused on essential charts"""
    
    def __init__(self, session_manager: SessionManager):
        super().__init__("Visualizations", session_manager)
    
    def render(self):
        """Render the simplified visualizations interface"""
        
        # Simple header
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0;">📈 Research Visualizations</h1>
            <p style="color: #f8f9fa; margin: 0.5rem 0 0 0;">Interactive charts and network analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check data availability
        if not self._check_data_availability():
            return
        
        # Simplified tabs - only the working ones
        tab1, tab2, tab3 = st.tabs([
            "🕸️ Network Analysis", 
            "📊 Research Charts", 
            "🔍 Data Explorer"
        ])
        
        with tab1:
            self._render_network_analysis()
        
        with tab2:
            self._render_research_charts()
        
        with tab3:
            self._render_data_explorer()
    
    def _check_data_availability(self):
        """Check if required data is available"""
        
        if not hasattr(st.session_state, 'papers') or not st.session_state.papers:
            self._render_no_papers_state()
            return False
        
        if not hasattr(st.session_state, 'analysis_results') or not st.session_state.analysis_results:
            self._render_no_analysis_state()
            return False
        
        return True
    
    def _render_no_papers_state(self):
        """No papers available state"""
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
                            <p style="color: #7d6608; margin-bottom: 0;">You can either collect new analysis or load from saved files.</p>
                        </div>
        """, unsafe_allow_html=True)
        
        # Two options: Collect new OR Load saved
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Collect New Papers")
            if st.button("🔍 Go to Data Collection", type="primary", use_container_width=True):
                self.session_manager.set_current_page('collection')
                st.rerun()
        
        with col2:
            st.markdown("### 📂 Load Saved Papers")
            self._render_papers_file_selector()
    
    def _render_papers_file_selector(self):
        """Papers file selection interface"""
        
        # Find papers files
        papers_files = glob.glob("data/raw/papers_*.json")
        
        if not papers_files:
            st.info("📁 No saved papers found")
            return
        
        # Format file list
        file_options = []
        file_paths = {}
        
        for file_path in papers_files:
            filename = os.path.basename(file_path)
            if filename.startswith('papers_'):
                timestamp_part = filename.replace('papers_', '').replace('.json', '')
                display_name = f"📄 {timestamp_part}"
                file_options.append(display_name)
                file_paths[display_name] = file_path
        
        if file_options:
            selected_display = st.selectbox(
                "Select papers file:",
                options=file_options,
                index=0
            )
            
            if st.button("📥 Load Papers", use_container_width=True):
                selected_file = file_paths[selected_display]
                self._load_papers_from_file(selected_file)

    def _load_papers_from_file(self, file_path):
        """Load papers from file"""
        
        try:
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
            
            st.success(f"✅ Loaded {len(papers)} papers from {os.path.basename(file_path)}")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error loading papers: {str(e)}")
    
    def _render_no_analysis_state(self):
        """No analysis results state"""
        
        papers_count = len(st.session_state.papers)
        
        st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #4ade80 0%, #22c55e 50%, #16a34a 100%);
                border: none;
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                margin-bottom: 1.5rem;
                box-shadow: 0 8px 32px rgba(34, 197, 94, 0.3);
                backdrop-filter: blur(10px);
            ">
                <h3 style="color: white; margin-bottom: 1rem; font-weight: 600;">🧠 Ready for Analysis!</h3>
                <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem;">You have <strong>{papers_count} papers</strong> ready for analysis.</p>
                <p style="color: rgba(255, 255, 255, 0.8);">You can run new analysis or load previous results.</p>
            </div>
            """, unsafe_allow_html=True)
        # Two options: Run new OR Load previous
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 Run New Analysis")
            if st.button("🧠 Go to Analysis Engine", type="primary"):
                self.session_manager.set_current_page('analysis')
                st.rerun()
        with col2:
            st.markdown("### 📂 Load Previous Results")
            self._render_analysis_file_selector()

        
    def _render_analysis_file_selector(self):
        """Analysis file selection interface"""
        # Find analysis result files
        analysis_files = glob.glob("data/processed/analysis_*.json")
        
        if not analysis_files:
            st.info("📁 No previous analysis results found")
            return
        
        # Format file list
        file_options = []
        file_paths = {}
        
        for file_path in analysis_files:
            filename = os.path.basename(file_path)
            if filename.startswith('analysis_'):
                timestamp_part = filename.replace('analysis_', '').replace('.json', '')
                display_name = f"📊 {timestamp_part}"
                file_options.append(display_name)
                file_paths[display_name] = file_path
        
        if file_options:
            selected_display = st.selectbox(
                "Select analysis results:",
                options=file_options,
                index=0
            )
            
            if st.button("📥 Load Results", use_container_width=True):
                selected_file = file_paths[selected_display]
                self._load_analysis_results(selected_file)

    def _load_analysis_results(self, file_path):
        """Load analysis results from file"""
        
        try:           
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract analysis results
            if 'results' in data:
                results = data['results']
            else:
                results = data
            
            # Save to session state
            st.session_state.analysis_results = results
            
            st.success(f"✅ Loaded analysis results from {os.path.basename(file_path)}")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error loading analysis results: {str(e)}")
    
    def _render_network_analysis(self):
        """Simplified network analysis"""
        
        st.subheader("🕸️ Author Collaboration Network")
        
        st.markdown("""
        **What this shows:** How researchers work together
        - **Each dot** = One author
        - **Each line** = Authors who collaborated  
        - **Dot size** = How many papers they wrote
        - **Color** = How many collaborators they have
        """)
        
        # Simple network type selection
        network_type = st.selectbox(
            "🔗 Network Type:",
            options=['Author Collaboration', 'Keyword Relationships'],
            help="Choose which type of network to visualize"
        )
        
        if network_type == 'Author Collaboration':
            self._render_author_network()
        else:
            self._render_keyword_network()
    
    def _create_real_author_collaboration_network(self):
        """REAL author collaboration analysis - PROFESSIONAL"""
        
        papers = st.session_state.papers
        results = st.session_state.analysis_results
        authors = results.get('authors', {})
        
        if not authors or not authors.get('most_productive'):
            return None
        
        # GET TOP AUTHORS
        most_productive = authors.get('most_productive', [])[:15]  # Top 15
        
        # REAL COLLABORATION MATRIX
        collaboration_count = defaultdict(int)
        author_stats = {}
        
        # Extract author names and their stats
        top_author_names = []
        for author_info in most_productive:
            if isinstance(author_info, (list, tuple)) and len(author_info) >= 2:
                author = author_info[0]
                stats = author_info[1]
            else:
                author = str(author_info)
                stats = {'paper_count': 1, 'collaboration_count': 0}
            
            top_author_names.append(author)
            author_stats[author] = stats
        
        # ANALYZE REAL CO-AUTHORSHIP FROM PAPERS
        for paper in papers:
            paper_authors = paper.get('authors', [])
            
            # Find which of our top authors are in this paper
            top_authors_in_paper = []
            for author in paper_authors:
                for top_author in top_author_names:
                    # Simple name matching (could be improved)
                    if top_author.lower() in author.lower() or author.lower() in top_author.lower():
                        top_authors_in_paper.append(top_author)
            
            # Create collaborations between authors in same paper
            for i, author1 in enumerate(top_authors_in_paper):
                for author2 in top_authors_in_paper[i+1:]:
                    # Count real collaborations
                    collaboration_count[(author1, author2)] += 1
                    collaboration_count[(author2, author1)] += 1
        
        # CREATE NETWORK BASED ON REAL COLLABORATIONS
        G = nx.Graph()
        
        # Add nodes (authors)
        for author in top_author_names:
            G.add_node(author, 
                    papers=author_stats[author].get('paper_count', 1),
                    collaborations=author_stats[author].get('collaboration_count', 0))
        
        # Add edges based on REAL co-authorship
        min_collaborations = 1  # Must have at least 1 paper together
        
        for (author1, author2), count in collaboration_count.items():
            if count >= min_collaborations and author1 != author2:
                G.add_edge(author1, author2, weight=count)
        
        return G

    def _render_author_network(self):
        """Create author collaboration network"""
        
        results = st.session_state.analysis_results
        authors = results.get('authors', {})
        
        if not authors or not authors.get('most_productive'):
            st.warning("🔍 No author data available for network.")
            return
        
        # Create network
        G = nx.Graph()
        
        # Add author nodes
        most_productive = authors.get('most_productive', [])[:10]  # Top 10 authors
        
        for author_info in most_productive:
            if isinstance(author_info, (list, tuple)) and len(author_info) >= 2:
                author = author_info[0]
                stats = author_info[1]
            else:
                author = str(author_info)
                stats = {'paper_count': 1, 'collaboration_count': 0}
            
            G.add_node(author, 
                      papers=stats.get('paper_count', 1),
                      collaborations=stats.get('collaboration_count', 0))
            
        # REAL collaboration analysis
        G = self._create_real_author_collaboration_network()

        if G is None or G.number_of_nodes() == 0:
            st.info("📊 No author collaboration data available.")
            return
    
        if G.number_of_nodes() == 0:
            st.info("📊 No network data to display.")
            return
        
       
        # Create visualization
        fig = self._create_simple_network(G, "Author Collaboration Network")
        
        st.write("🔍 DEBUG - Network Analysis:")
        st.write(f"Total nodes: {G.number_of_nodes()}")
        st.write(f"Total edges: {G.number_of_edges()}")
        
        st.write("Node connections:")
        for node in G.nodes():
            connections = list(G.neighbors(node))
            degree = G.degree(node)
            st.write(f"- {node}: {degree} connections → {connections}")

        # Isolated nodes
        isolated = list(nx.isolates(G))
        if isolated:
            st.write(f"⚠️ Isolated nodes: {isolated}")

        st.plotly_chart(fig, use_container_width=True)
        # Simple network stats
        self._show_network_stats(G, "keyword")
        self._show_network_stats(G, "author")
        
    def _render_keyword_network(self):
        """Create keyword relationships network"""
        
        results = st.session_state.analysis_results
        keywords = results.get('global_keywords', [])
        
        if not keywords or len(keywords) < 5:
            st.warning("🔍 Need more keywords for network visualization.")
            return
        
        # Create keyword network
        G = nx.Graph()
        
        # Add top keywords as nodes
        top_keywords = [kw[0] for kw in keywords[:12]]  # Top 12 keywords
        keyword_freq = {kw[0]: kw[1] for kw in keywords[:12]}
        
        # VERIFICATION TESTS
        # st.write("---DEBUG VERIFICATION---")
        # st.write("1. Keywords source:", results.get('global_keywords', 'NOT FOUND')[:3])

        # st.write("2. Sample paper abstracts:")
        # papers = st.session_state.papers
        # for i, paper in enumerate(papers[:2]):
        #     abstract = paper.get('abstract', 'No abstract')[:150]
        #     st.write(f"Paper {i+1}: {abstract}...")

        # st.write("3. Manual co-occurrence test:")
        # count = 0
        # for paper in papers:
        #     text = paper.get('abstract', '').lower()
        #     if 'intelligence' in text and 'artificial' in text:
        #         count += 1
        # st.write(f"Papers with both 'intelligence' AND 'artificial': {count}")
        # st.write("---END DEBUG---")
        
        for keyword in top_keywords:
            G.add_node(keyword, frequency=keyword_freq[keyword])
        
        # Connect related keywordsf_render_keyword_network 
        for i, kw1 in enumerate(top_keywords):
            for j, kw2 in enumerate(top_keywords[i+1:], i+1):
                
                G = self._create_real_keyword_cooccurrence_network()

                if G is None or G.number_of_nodes() == 0:
                    st.info("📊 No keyword co-occurrences found.")
                    return
        
        if G.number_of_edges() == 0:
            st.info("📊 No keyword relationships found.")
            return
        
        # Create visualization
        fig = self._create_keyword_network_viz(G)
        st.plotly_chart(fig, use_container_width=True)
        
        # Keyword insights
        self._show_keyword_insights(G)
        
    def _create_real_keyword_cooccurrence_network(self):
        """REAL keyword co-occurrence analysis - PROFESSIONAL"""
        
        papers = st.session_state.papers
        results = st.session_state.analysis_results
        keywords = results.get('global_keywords', [])
        
        if len(keywords) < 5:
            st.warning("Need more keywords for analysis")
            return
        
        # GET TOP KEYWORDS
        top_keywords = [kw[0] for kw in keywords[:15]]  # Top 15 real keywords
        
        # CALCULATE CO-OCCURRENCE MATRIX
        cooccurrence_matrix = {}
        
        for kw1 in top_keywords:
            for kw2 in top_keywords:
                if kw1 != kw2:
                    # COUNT PAPERS WHERE BOTH KEYWORDS APPEAR
                    cooccur_count = 0
                    for paper in papers:
                        abstract = paper.get('abstract', '').lower()
                        title = paper.get('title', '').lower()
                        text = abstract + " " + title
                        
                        if kw1.lower() in text and kw2.lower() in text:
                            cooccur_count += 1
                    
                    cooccurrence_matrix[(kw1, kw2)] = cooccur_count
            
            # CREATE NETWORK BASED ON REAL CO-OCCURRENCE
            G = nx.Graph()
            
            # Add nodes (keywords)
            keyword_freq = {kw[0]: kw[1] for kw in keywords[:15]}
            for keyword in top_keywords:
                G.add_node(keyword, frequency=keyword_freq[keyword])
            
            # Add edges based on REAL co-occurrence (minimum threshold)
            min_cooccurrence = 2  # Must appear together in at least 2 papers
            
            for (kw1, kw2), count in cooccurrence_matrix.items():
                if count >= min_cooccurrence:
                    G.add_edge(kw1, kw2, weight=count)
            
            return G
    def _create_simple_network(self, G, title):
            """Create simple network visualization"""
            
            # Calculate layout
            pos = nx.spring_layout(G, k=1.5, iterations=50)
            
            # Extract edges
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='rgba(150,150,150,0.5)'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Extract nodes
            node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node info
                node_attrs = G.nodes[node]
                if 'papers' in node_attrs:  # Author network
                    papers = node_attrs['papers']
                    collabs = node_attrs['collaborations']
                    node_text.append(f"{node}<br>Papers: {papers}<br>Collaborations: {collabs}")
                    node_size.append(max(15, papers * 4))
                    node_color.append(collabs)
                else:  # Keyword network
                    freq = node_attrs.get('frequency', 1)
                    connections = len(list(G.neighbors(node)))
                    node_text.append(f"{node}<br>Frequency: {freq}<br>Connections: {connections}")
                    node_size.append(max(15, freq * 3))
                    node_color.append(freq)
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hovertemplate='%{customdata}<extra></extra>',
                text=[node[:15] + '...' if len(node) > 15 else node for node in G.nodes()],
                textposition="middle center",
                customdata=node_text,
                marker=dict(
                    size=node_size,
                    color=node_color,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Frequency"),
                    line=dict(width=2, color='white')
                )
            )
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(text="🔗 Keyword Co-occurrence Network", font=dict(size=16)),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[
                        dict(
                            text=f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} connections",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(size=12)
                        )
                    ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=False),
                    dragmode='pan',
                    template='plotly_dark',
                    # scrollZoom=True,
                    # doubleClick='reset+autosize'  # Double click ile zoom reset
                )
            )
                    
            return fig
    
    def _create_keyword_network_viz(self, G):
        """Create keyword network visualization"""
        
        # Same as _create_simple_network but optimized for keywords
        return self._create_simple_network(G, "🔗 Keyword Relationships Network")
    
    def _show_network_stats(self, G, network_type="author"):
        """Show simple network statistics"""
        
        st.markdown("#### 📊 Network Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # st.metric("Authors", G.number_of_nodes())
            if network_type == "keyword":
                st.metric("Keywords", G.number_of_nodes())
            else:
                st.metric("Authors", G.number_of_nodes())
        
        with col2:
            st.metric("Connections", G.number_of_edges())
        
        with col3:
            if G.number_of_nodes() > 0:
                density = nx.density(G)
                st.metric("Network Density", f"{density:.3f}")
            else:
                st.metric("Network Density", "0")
        
        with col4:
            if G.number_of_nodes() > 0:
                components = nx.number_connected_components(G)
                st.metric("Groups", components)
            else:
                st.metric("Groups", "0")
        
        # Explanations
        with st.expander("ℹ️ What do these numbers mean?"):
            st.markdown("""
            - **Authors/Keywords**: Number of researchers or terms in the network
            - **Connections**: How many collaboration links exist
            - **Network Density**: How connected everyone is (0=no connections, 1=everyone connected)
            - **Groups**: How many separate research communities exist
            """)
    
    def _show_keyword_insights(self, G):
        """Show keyword network insights"""
        
        st.markdown("#### 💡 Keyword Insights")
        
        if G.number_of_nodes() > 0:
            # Most connected keyword
            centrality = nx.degree_centrality(G)
            most_connected = max(centrality.items(), key=lambda x: x[1])
            
            col1, col2 = st.columns(2)
            
            with col1:
                connections = len(list(G.neighbors(most_connected[0])))
                st.metric("Most Connected Term", most_connected[0], f"{connections} connections")
            
            with col2:
                avg_connections = sum(dict(G.degree()).values()) / G.number_of_nodes()
                st.metric("Avg Connections", f"{avg_connections:.1f}")
    
    def _render_research_charts(self):
        """Simplified research charts"""
        
        st.subheader("📊 Research Overview Charts")
        
        results = st.session_state.analysis_results
        
        # Chart selection
        chart_type = st.selectbox(
            "📈 Choose Chart:",
            options=['Cluster Distribution', 'Top Keywords', 'Author Productivity'],
            help="Select which aspect of your research to visualize"
        )
        
        if chart_type == 'Cluster Distribution':
            self._render_cluster_charts()
        elif chart_type == 'Top Keywords':
            self._render_keyword_charts()
        else:
            self._render_author_charts()
    
    def _render_cluster_charts(self):
        """Simple cluster visualization"""
        
        results = st.session_state.analysis_results
        clusters = results.get('clustering', {}).get('clusters', {})
        
        if not clusters:
            st.warning("🔍 No clustering data available.")
            return
        
        # Prepare data
        cluster_data = []
        for cluster_id, cluster_info in clusters.items():
            size = cluster_info.get('size', 0)
            name = cluster_info.get('name', 'Uncategorized' if cluster_id == '-1' else f'Cluster {cluster_id}')
            cluster_data.append({'Cluster': name, 'Papers': size})
        
        df = pd.DataFrame(cluster_data)
        
        # Show table first
        st.dataframe(df, use_container_width=True)

        cluster_names = df['Cluster'].tolist()
        paper_counts = df['Papers'].tolist()

        # Side by side charts
        col1, col2 = st.columns(2)

        with col1:
            fig_pie = go.Figure(data=[go.Pie(
                labels=cluster_names, 
                values=paper_counts,
                hole=0.3
            )])
            fig_pie.update_layout(title="📊 Research Areas Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_bar = go.Figure(data=[go.Bar(
                x=cluster_names,
                y=paper_counts,
                marker=dict(color=paper_counts, colorscale='Blues')
            )])
            fig_bar.update_layout(
                title="📈 Papers per Research Area",
                xaxis_title="Clusters",
                yaxis_title="Papers",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def _render_keyword_charts(self):
        """Simple keyword visualization"""
        
        results = st.session_state.analysis_results
        keywords = results.get('global_keywords', [])
        
        if not keywords:
            st.warning("🔍 No keyword data available.")
            return
        
        # Top 12 keywords
        top_keywords = keywords[:12]
        df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
        
        # Data table first
        st.dataframe(df, use_container_width=True)

        keywords_list = df['Keyword'].tolist()
        frequency_list = df['Frequency'].tolist()

        fig = go.Figure(data=[
            go.Bar(
                y=keywords_list,
                x=frequency_list,
                orientation='h',
                marker=dict(color=frequency_list, colorscale='Viridis')
            )
        ])

        fig.update_layout(
            title='🔑 Most Frequent Research Keywords',
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)
    
    def _render_author_charts(self):
        """Simple author productivity visualization"""
        
        results = st.session_state.analysis_results
        authors = results.get('authors', {})
        most_productive = authors.get('most_productive', [])
        
        if not most_productive:
            st.warning("🔍 No author data available.")
            return
        
        # Prepare author data
        author_data = []
        for item in most_productive[:10]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                author = item[0]
                stats = item[1]
            else:
                author = str(item)
                stats = {'paper_count': 1, 'collaboration_count': 0}
            
            author_data.append({
                'Author': author[:25] + '...' if len(author) > 25 else author,
                'Papers': stats.get('paper_count', 1),
                'Collaborations': stats.get('collaboration_count', 0)
            })
        df = pd.DataFrame(author_data)
        
        # Show table first
        st.markdown("#### 🏆 Top Authors")
        st.dataframe(df, use_container_width=True)

        # SHEET'TEN VERİYİ AL VE PLOT YAP
        authors_list = df['Author'].tolist()
        papers_list = df['Papers'].tolist()
        collabs_list = df['Collaborations'].tolist()

        # Bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=authors_list,
            y=papers_list,
            name='Papers',
            marker_color='lightblue'
        ))

        fig.add_trace(go.Bar(
            x=authors_list,
            y=collabs_list,
            name='Collaborations',
            marker_color='orange'
        ))

        fig.update_layout(
            title='👥 Author Papers & Collaborations',
            xaxis_title='Authors',
            yaxis_title='Count',
            barmode='group',
            xaxis_tickangle=-45
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_data_explorer(self):
        """Simple data explorer"""
        
        st.subheader("🔍 Explore Your Research Data")
        
        papers = st.session_state.papers
        results = st.session_state.analysis_results
        
        # Simple filters
        st.markdown("#### 🎛️ Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Keyword filter
            keywords = results.get('global_keywords', [])
            keyword_options = ['All'] + [kw[0] for kw in keywords[:15]]
            selected_keyword = st.selectbox("🔑 Filter by Keyword", keyword_options)
        
        with col2:
            # Author filter  
            authors = results.get('authors', {})
            most_productive = authors.get('most_productive', [])
            author_options = ['All'] + [item[0] if isinstance(item, (list, tuple)) else str(item) 
                                       for item in most_productive[:10]]
            selected_author = st.selectbox("👥 Filter by Author", author_options)
        
        # Apply filters
        filtered_papers = papers.copy()
        
        if selected_keyword != 'All':
            filtered_papers = [p for p in filtered_papers 
                             if selected_keyword.lower() in p.get('abstract', '').lower()]
        
        if selected_author != 'All':
            filtered_papers = [p for p in filtered_papers 
                             if any(selected_author.lower() in author.lower() 
                                   for author in p.get('authors', []))]
        
        # Show results
        st.markdown("#### 📊 Filtered Results")
        st.success(f"📄 Found {len(filtered_papers)} papers matching your filters")
        
        # Show sample papers
        if filtered_papers:
            for i, paper in enumerate(filtered_papers[:3], 1):
                with st.expander(f"📄 Paper {i}: {paper.get('title', 'Untitled')[:50]}..."):
                    st.markdown(f"**Title:** {paper.get('title', 'N/A')}")
                    st.markdown(f"**Authors:** {', '.join(paper.get('authors', ['Unknown']))}")
                    st.markdown(f"**Abstract:** {paper.get('abstract', 'N/A')[:300]}...")


# Export for main dashboard
__all__ = ['VisualizationsComponent']