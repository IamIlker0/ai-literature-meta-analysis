"""
NLP Analysis Pipeline for Scientific Literature Meta-Analysis
Provides semantic analysis, clustering, and insight extraction capabilities
"""

import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime

# NLP Libraries
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class LiteratureNLPAnalyzer:
    """Comprehensive NLP analysis for scientific literature"""
    
    def __init__(self, config=None):
        """Initialize NLP analyzer with models and settings"""
        
        print("🧠 Initializing NLP Analysis Pipeline...")
        
        # Load sentence transformer model for semantic analysis
        print("📥 Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add scientific stop words
        self.scientific_stop_words = {
            'paper', 'research', 'study', 'method', 'approach', 'technique',
            'algorithm', 'model', 'framework', 'system', 'analysis', 'work',
            'result', 'conclusion', 'introduction', 'related', 'proposed',
            'show', 'present', 'demonstrate', 'evaluate', 'compare', 'improve'
        }
        
        self.all_stop_words = self.stop_words.union(self.scientific_stop_words)
        
        # Analysis settings
        self.similarity_threshold = 0.7
        self.clustering_eps = 0.3
        self.min_cluster_size = 2
        
        print("✅ NLP Pipeline initialized successfully!")
    
    def load_papers_from_file(self, filepath: str) -> List[Dict]:
        """Load papers from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            print(f"📄 Loaded {len(papers)} papers from {filepath}")
            return papers
        except Exception as e:
            print(f"❌ Error loading papers: {e}")
            return []
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, int]]:
        """Extract important keywords from text using frequency analysis"""
        
        # Clean and tokenize text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text)
        
        tokens = word_tokenize(text)
        
        # Filter tokens
        keywords = []
        for token in tokens:
            if (len(token) > 3 and 
                token not in self.all_stop_words and 
                token.isalpha()):
                # Lemmatize
                lemmatized = self.lemmatizer.lemmatize(token)
                keywords.append(lemmatized)
        
        # Count frequencies
        keyword_counts = Counter(keywords)
        return keyword_counts.most_common(top_k)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate semantic embeddings for texts using sentence transformer"""
        print("🔄 Generating semantic embeddings...")
        
        # Clean texts
        cleaned_texts = []
        for text in texts:
            # Remove excessive whitespace and normalize
            cleaned = re.sub(r'\s+', ' ', text).strip()
            cleaned_texts.append(cleaned)
        
        # Generate embeddings
        embeddings = self.sentence_model.encode(cleaned_texts, show_progress_bar=True)
        
        print(f"✅ Generated embeddings: {embeddings.shape}")
        return embeddings
    
    def cluster_papers(self, papers: List[Dict], method: str = 'dbscan') -> Dict:
        """Cluster papers based on abstract similarity"""
        
        print(f"🔄 Clustering papers using {method.upper()}...")
        
        # Extract abstracts
        abstracts = [paper['abstract'] for paper in papers]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(abstracts)
        
        # Apply clustering
        if method == 'dbscan':
            clusterer = DBSCAN(eps=self.clustering_eps, min_samples=self.min_cluster_size, metric='cosine')
            cluster_labels = clusterer.fit_predict(embeddings)
        elif method == 'kmeans':
            n_clusters = min(4, len(papers) // 2)  # Adaptive cluster count
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Organize results
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append({
                'paper': papers[i],
                'embedding': embeddings[i]
            })
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id, cluster_papers in clusters.items():
            if cluster_id == -1:  # Noise cluster in DBSCAN
                cluster_name = "Uncategorized"
            else:
                # Extract common keywords for cluster naming
                all_abstracts = " ".join([cp['paper']['abstract'] for cp in cluster_papers])
                keywords = self.extract_keywords(all_abstracts, top_k=3)
                cluster_name = ", ".join([kw[0] for kw in keywords])
            
            cluster_stats[cluster_id] = {
                'name': cluster_name,
                'size': len(cluster_papers),
                'papers': [cp['paper'] for cp in cluster_papers]
            }
        
        print(f"✅ Clustering complete: {len(cluster_stats)} clusters found")
        
        return {
            'clusters': cluster_stats,
            'embeddings': embeddings,
            'method': method
        }
    
    def find_similar_papers(self, target_paper: Dict, all_papers: List[Dict], 
                          top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Find papers most similar to target paper"""
        
        print(f"🔍 Finding papers similar to: '{target_paper['title'][:50]}...'")
        
        # Generate embeddings for all abstracts
        all_abstracts = [paper['abstract'] for paper in all_papers]
        target_abstract = target_paper['abstract']
        
        # Get embeddings
        all_embeddings = self.generate_embeddings(all_abstracts)
        target_embedding = self.generate_embeddings([target_abstract])
        
        # Calculate similarities
        similarities = cosine_similarity(target_embedding, all_embeddings)[0]
        
        # Get top similar papers (excluding self)
        similar_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in similar_indices:
            if all_papers[idx]['id'] != target_paper['id']:  # Exclude self
                similarity_score = similarities[idx]
                if similarity_score > self.similarity_threshold:
                    results.append((all_papers[idx], similarity_score))
                    if len(results) >= top_k:
                        break
        
        print(f"✅ Found {len(results)} similar papers")
        return results
    
    def analyze_author_networks(self, papers: List[Dict]) -> Dict:
        """Analyze author collaboration networks"""
        
        print("🔄 Analyzing author collaboration networks...")
        
        # Extract all authors and their collaborations
        author_papers = defaultdict(list)
        author_collaborations = defaultdict(set)
        
        for paper in papers:
            authors = paper['authors']
            paper_info = {
                'title': paper['title'],
                'date': paper['date'],
                'categories': paper['categories']
            }
            
            # Record papers for each author
            for author in authors:
                author_papers[author].append(paper_info)
            
            # Record collaborations
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    author_collaborations[author1].add(author2)
                    author_collaborations[author2].add(author1)
        
        # Calculate author statistics
        author_stats = {}
        for author, paper_list in author_papers.items():
            stats = {
                'paper_count': len(paper_list),
                'papers': paper_list,
                'collaborators': list(author_collaborations[author]),
                'collaboration_count': len(author_collaborations[author]),
                'categories': list(set(cat for paper in paper_list for cat in paper['categories']))
            }
            author_stats[author] = stats
        
        # Find most productive and collaborative authors
        most_productive = sorted(author_stats.items(), key=lambda x: x[1]['paper_count'], reverse=True)
        most_collaborative = sorted(author_stats.items(), key=lambda x: x[1]['collaboration_count'], reverse=True)
        
        print(f"✅ Author network analysis complete: {len(author_stats)} unique authors")
        
        return {
            'author_stats': author_stats,
            'most_productive': most_productive[:5],
            'most_collaborative': most_collaborative[:5],
            'total_authors': len(author_stats),
            'total_collaborations': sum(len(collabs) for collabs in author_collaborations.values()) // 2
        }
    
    def analyze_research_trends(self, papers: List[Dict]) -> Dict:
        """Analyze research trends over time"""
        
        print("🔄 Analyzing research trends...")
        
        # Group papers by year
        papers_by_year = defaultdict(list)
        for paper in papers:
            try:
                year = int(paper['date'][:4]) if paper['date'] else 2023
                papers_by_year[year].append(paper)
            except (ValueError, IndexError):
                papers_by_year[2023].append(paper)  # Default year
        
        # Extract keywords by year
        yearly_trends = {}
        for year, year_papers in papers_by_year.items():
            # Combine all abstracts and titles for the year
            year_text = " ".join([paper['title'] + " " + paper['abstract'] for paper in year_papers])
            
            # Extract top keywords
            keywords = self.extract_keywords(year_text, top_k=10)
            
            # Extract categories
            categories = []
            for paper in year_papers:
                categories.extend(paper['categories'])
            category_counts = Counter(categories)
            
            yearly_trends[year] = {
                'paper_count': len(year_papers),
                'top_keywords': keywords,
                'top_categories': category_counts.most_common(5),
                'papers': year_papers
            }
        
        print(f"✅ Trend analysis complete: {len(yearly_trends)} years analyzed")
        
        return {
            'yearly_trends': yearly_trends,
            'total_years': len(yearly_trends),
            'year_range': f"{min(yearly_trends.keys())}-{max(yearly_trends.keys())}"
        }
    
    def comprehensive_analysis(self, papers: List[Dict]) -> Dict:
        """Perform comprehensive analysis on paper collection"""
        
        print(f"\n🔬 COMPREHENSIVE NLP ANALYSIS")
        print("=" * 60)
        print(f"📊 Analyzing {len(papers)} papers...")
        
        results = {
            'metadata': {
                'total_papers': len(papers),
                'analysis_date': datetime.now().isoformat(),
                'papers_analyzed': [p['title'] for p in papers]
            }
        }
        
        # 1. Clustering Analysis
        print(f"\n1️⃣ CLUSTERING ANALYSIS")
        clustering_results = self.cluster_papers(papers, method='dbscan')
        results['clustering'] = clustering_results
        
        # 2. Author Network Analysis
        print(f"\n2️⃣ AUTHOR NETWORK ANALYSIS")
        author_analysis = self.analyze_author_networks(papers)
        results['authors'] = author_analysis
        
        # 3. Research Trends Analysis
        print(f"\n3️⃣ RESEARCH TRENDS ANALYSIS")
        trends_analysis = self.analyze_research_trends(papers)
        results['trends'] = trends_analysis
        
        # 4. Global Keywords Analysis
        print(f"\n4️⃣ GLOBAL KEYWORDS ANALYSIS")
        all_text = " ".join([paper['title'] + " " + paper['abstract'] for paper in papers])
        global_keywords = self.extract_keywords(all_text, top_k=20)
        results['global_keywords'] = global_keywords
        
        print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        
        return results


# Test function for the NLP pipeline
def test_nlp_pipeline():
    """Test NLP pipeline with saved paper data"""
    
    print("🧪 TESTING NLP ANALYSIS PIPELINE")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = LiteratureNLPAnalyzer()
    
    # Find the most recent paper file
    import glob
    import os
    
    # Look for JSON files in data/raw
    json_files = glob.glob("data/raw/ai_papers_*.json")
    if not json_files:
        json_files = glob.glob("data/raw/arxiv_papers_*.json")
    
    if not json_files:
        print("❌ No paper data files found!")
        print("Please run the ArXiv collector first: python test_arxiv.py")
        return
    
    # Use the most recent file
    latest_file = max(json_files, key=os.path.getctime)
    print(f"📄 Using data from: {latest_file}")
    
    # Load papers
    papers = analyzer.load_papers_from_file(latest_file)
    
    if not papers:
        print("❌ No papers loaded!")
        return
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_analysis(papers)
    
    # Display results summary
    print(f"\n📊 ANALYSIS RESULTS SUMMARY")
    print("=" * 50)
    
    # Clustering results
    clusters = results['clustering']['clusters']
    print(f"🔗 Clusters found: {len(clusters)}")
    for cluster_id, cluster_info in clusters.items():
        if cluster_id != -1:
            print(f"   Cluster {cluster_id}: {cluster_info['name']} ({cluster_info['size']} papers)")
    
    # Author insights
    authors = results['authors']
    print(f"👥 Total authors: {authors['total_authors']}")
    print(f"🤝 Collaborations: {authors['total_collaborations']}")
    if authors['most_productive']:
        top_author = authors['most_productive'][0]
        print(f"📝 Most productive: {top_author[0]} ({top_author[1]['paper_count']} papers)")
    
    # Trend insights
    trends = results['trends']
    print(f"📈 Years analyzed: {trends['year_range']}")
    
    # Global keywords
    keywords = results['global_keywords']
    print(f"🔑 Top keywords: {', '.join([kw[0] for kw in keywords[:5]])}")
    
    # Save results
    output_file = f"data/processed/nlp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert numpy arrays to lists for JSON serialization
        import json
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
       # Simplified results for JSON (remove embeddings)
        json_results = {k: v for k, v in results.items() if k != 'clustering'}
        
        # Fix clustering section - convert numpy int64 keys to strings
        clustering_fixed = {}
        for k, v in results['clustering'].items():
            if k != 'embeddings':
                if k == 'clusters':
                    # Convert cluster keys from numpy int64 to strings
                    clusters_fixed = {}
                    for cluster_id, cluster_info in v.items():
                        clusters_fixed[str(cluster_id)] = cluster_info
                    clustering_fixed[k] = clusters_fixed
                else:
                    clustering_fixed[k] = v
        
        json_results['clustering'] = clustering_fixed
        
        json.dump(json_results, f, indent=2, default=convert_numpy, ensure_ascii=False)
    print(f"💾 Analysis results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    # Run test when file is executed directly
    test_nlp_pipeline()