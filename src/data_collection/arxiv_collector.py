"""
ArXiv Academic Paper Collector
Collects research papers from ArXiv repository using their public API
"""

import requests
import xml.etree.ElementTree as ET
import time
import re
from typing import List, Dict, Optional
from datetime import datetime

class ArXivCollector:
    """Collects research papers from ArXiv repository"""
    
    def __init__(self, config):
        self.base_url = config.ARXIV_BASE_URL
        self.delay = config.ARXIV_DELAY
        self.max_papers = config.get_max_papers()
        
        print(f"🔬 ArXiv Collector initialized")
        print(f"📊 Max papers per search: {self.max_papers}")
        print(f"⏱️ Request delay: {self.delay} seconds")
    
    def search_papers(self, query: str, max_results: int = None, 
                     category: str = None) -> List[Dict]:
        """
        Search and collect papers from ArXiv
        
        Args:
            query: Search query (e.g., "machine learning", "transformer neural network")
            max_results: Maximum papers to collect (uses config default if None)
            category: ArXiv category filter (e.g., "cs.AI", "cs.LG")
            
        Returns:
            List of paper dictionaries with metadata
        """
        if max_results is None:
            max_results = self.max_papers
            
        print(f"\n🔍 Searching ArXiv for: '{query}'")
        print(f"📄 Collecting up to {max_results} papers...")
        
        # Format query for ArXiv API
        formatted_query = self._format_query(query, category)
        
        # Build request parameters
        params = {
            'search_query': formatted_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            # Make API request
            print("📡 Making API request to ArXiv...")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            papers = self._parse_arxiv_response(response.content)
            
            print(f"✅ Successfully collected {len(papers)} papers from ArXiv")
            
            # Respect rate limits
            if len(papers) > 0:
                print(f"⏱️ Waiting {self.delay} seconds (rate limiting)...")
                time.sleep(self.delay)
            
            return papers
            
        except requests.RequestException as e:
            print(f"❌ ArXiv API error: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return []
    
    def _format_query(self, query: str, category: str = None) -> str:
        """Format query string for ArXiv API"""
        # Clean and format the query
        cleaned_query = re.sub(r'[^\w\s]', ' ', query)  # Remove special chars
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()  # Normalize spaces
        
        # ArXiv query format
        formatted = f"all:{cleaned_query}"
        
        # Add category filter if specified
        if category:
            formatted += f" AND cat:{category}"
        
        return formatted
    
    def _parse_arxiv_response(self, xml_content: bytes) -> List[Dict]:
        """Parse ArXiv XML response into structured paper data"""
        try:
            root = ET.fromstring(xml_content)
            
            # XML namespaces for ArXiv API
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            papers = []
            
            # Process each paper entry
            for entry in root.findall('atom:entry', namespaces):
                paper = self._extract_paper_info(entry, namespaces)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except ET.ParseError as e:
            print(f"❌ XML parsing error: {e}")
            return []
        except Exception as e:
            print(f"❌ Response parsing error: {e}")
            return []
    
    def _extract_paper_info(self, entry, namespaces) -> Optional[Dict]:
        """Extract structured information from a single ArXiv paper entry"""
        try:
            # Extract title
            title_elem = entry.find('atom:title', namespaces)
            title = title_elem.text.strip() if title_elem is not None else "Unknown Title"
            title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
            
            # Extract abstract
            summary_elem = entry.find('atom:summary', namespaces)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            abstract = re.sub(r'\s+', ' ', abstract)  # Normalize whitespace
            
            # Extract ArXiv ID
            id_elem = entry.find('atom:id', namespaces)
            arxiv_url = id_elem.text if id_elem is not None else ""
            arxiv_id = arxiv_url.split('/')[-1] if arxiv_url else ""
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name_elem = author.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Extract publication date
            published_elem = entry.find('atom:published', namespaces)
            date = published_elem.text[:10] if published_elem is not None else ""
            
            # Extract updated date
            updated_elem = entry.find('atom:updated', namespaces)
            updated = updated_elem.text[:10] if updated_elem is not None else date
            
            # Extract categories/subjects
            categories = []
            for category in entry.findall('atom:category', namespaces):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Extract PDF link
            pdf_url = ""
            for link in entry.findall('atom:link', namespaces):
                if link.get('type') == 'application/pdf':
                    pdf_url = link.get('href', '')
                    break
            
            # Calculate paper age (for relevance)
            try:
                from datetime import datetime
                pub_date = datetime.strptime(date, '%Y-%m-%d') if date else datetime.now()
                days_old = (datetime.now() - pub_date).days
            except:
                days_old = 0
            
            # Return structured paper data
            return {
                'id': f"arxiv_{arxiv_id}",
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'date': date,
                'updated': updated,
                'source': 'arxiv',
                'arxiv_id': arxiv_id,
                'arxiv_url': arxiv_url,
                'categories': categories,
                'pdf_url': pdf_url,
                'venue': 'ArXiv Preprint',
                'citation_count': 0,  # ArXiv doesn't provide citation counts
                'days_old': days_old,
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error extracting paper info: {e}")
            return None
    
    def get_popular_categories(self) -> Dict[str, str]:
        """Get popular ArXiv categories for filtering"""
        return {
            'cs.AI': 'Artificial Intelligence',
            'cs.LG': 'Machine Learning', 
            'cs.CL': 'Computation and Language (NLP)',
            'cs.CV': 'Computer Vision',
            'cs.IR': 'Information Retrieval',
            'stat.ML': 'Statistics - Machine Learning',
            'q-bio.QM': 'Quantitative Biology',
            'physics.med-ph': 'Medical Physics'
        }
    
    def search_by_category(self, category: str, max_results: int = None) -> List[Dict]:
        """Search papers by specific ArXiv category"""
        if max_results is None:
            max_results = self.max_papers
            
        categories = self.get_popular_categories()
        category_name = categories.get(category, category)
        
        print(f"\n🏷️ Searching category: {category} ({category_name})")
        
        # Use category as main search criterion
        formatted_query = f"cat:{category}"
        
        params = {
            'search_query': formatted_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',  # Recent papers first
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            papers = self._parse_arxiv_response(response.content)
            print(f"✅ Found {len(papers)} papers in {category_name}")
            
            time.sleep(self.delay)
            return papers
            
        except requests.RequestException as e:
            print(f"❌ Category search error: {e}")
            return []


# Test function for quick validation
def test_arxiv_collector():
    """Quick test of ArXiv collector functionality"""
    print("🧪 Testing ArXiv Collector...")
    print("=" * 50)
    
    # Import config
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from src.config.settings import Config
        config = Config()
        
        # Initialize collector
        collector = ArXivCollector(config)
        
        # Test search
        papers = collector.search_papers("machine learning", max_results=5)
        
        if papers:
            print(f"\n✅ SUCCESS! Found {len(papers)} papers")
            print("\n📄 Sample Paper:")
            print("-" * 40)
            sample = papers[0]
            print(f"Title: {sample['title']}")
            print(f"Authors: {', '.join(sample['authors'][:3])}...")
            print(f"Date: {sample['date']}")
            print(f"Categories: {', '.join(sample['categories'][:2])}")
            print(f"Abstract: {sample['abstract'][:150]}...")
            print(f"ArXiv ID: {sample['arxiv_id']}")
            
            return papers
        else:
            print("❌ No papers found")
            return []
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return []


if __name__ == "__main__":
    # Run test when file is executed directly
    test_arxiv_collector()