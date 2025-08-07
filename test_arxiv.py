import sys
import json
from datetime import datetime
sys.path.append('.')

from src.config.settings import Config
from src.data_collection.arxiv_collector import ArXivCollector
import json
import pandas as pd
import os
from datetime import datetime
sys.path.append('.')

from src.config.settings import Config
from src.data_collection.arxiv_collector import ArXivCollector

def save_papers_multiple_formats(papers, query_name="arxiv_papers"):
    """Save papers in both JSON and CSV formats"""
    
    if not papers:
        print("❌ No papers to save")
        return
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Ensure directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Save as JSON (raw format)
    json_filename = f"data/raw/{query_name}_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    
    # Save as CSV (processed format)
    csv_filename = f"data/processed/{query_name}_{timestamp}.csv"
    
    # Prepare data for CSV (flatten complex fields)
    csv_data = []
    for paper in papers:
        csv_row = {
            'id': paper['id'],
            'title': paper['title'],
            'first_author': paper['authors'][0] if paper['authors'] else 'Unknown',
            'all_authors': '; '.join(paper['authors']),
            'author_count': len(paper['authors']),
            'date': paper['date'],
            'source': paper['source'],
            'arxiv_id': paper['arxiv_id'],
            'main_category': paper['categories'][0] if paper['categories'] else 'Unknown',
            'all_categories': '; '.join(paper['categories']),
            'abstract_length': len(paper['abstract']),
            'abstract': paper['abstract'][:500] + '...' if len(paper['abstract']) > 500 else paper['abstract'],
            'pdf_url': paper['pdf_url'],
            'days_old': paper['days_old'],
            'collected_at': paper['collected_at']
        }
        csv_data.append(csv_row)
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    
    print(f"💾 Papers saved in multiple formats:")
    print(f"   📄 JSON: {json_filename}")
    print(f"   📊 CSV:  {csv_filename}")
    
    return json_filename, csv_filename

def display_paper_summary(papers):
    """Display a nice summary of collected papers"""
    
    if not papers:
        print("❌ No papers to display")
        return
    
    print(f"\n📚 PAPER COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total Papers: {len(papers)}")
    
    # Author statistics
    all_authors = []
    for paper in papers:
        all_authors.extend(paper['authors'])
    
    print(f"Unique Authors: {len(set(all_authors))}")
    print(f"Total Author Mentions: {len(all_authors)}")
    
    # Date range
    dates = [paper['date'] for paper in papers if paper['date']]
    if dates:
        print(f"Date Range: {min(dates)} to {max(dates)}")
    
    # Categories
    all_categories = []
    for paper in papers:
        all_categories.extend(paper['categories'])
    
    category_counts = {}
    for cat in all_categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nTop Categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {cat}: {count} papers")
    
    print(f"\n📄 PAPER LIST:")
    print("-" * 60)
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
        print(f"   Date: {paper['date']} | Category: {paper['categories'][0] if paper['categories'] else 'Unknown'}")
        print(f"   ArXiv ID: {paper['arxiv_id']}")
        print(f"   Abstract: {paper['abstract'][:150]}...")

def test_multiple_queries():
    """Test collector with multiple different queries"""
    
    print("🧪 COMPREHENSIVE ArXiv COLLECTOR TEST")
    print("=" * 70)
    
    config = Config()
    collector = ArXivCollector(config)
    
    # Test different queries
    test_queries = [
        ("machine_learning", "machine learning"),
        ("transformer_models", "transformer neural network"),
        ("computer_vision", "computer vision deep learning")
    ]
    
    all_papers = []
    
    for query_name, query_text in test_queries:
        print(f"\n🔍 Testing Query: '{query_text}'")
        print("-" * 50)
        
        papers = collector.search_papers(query_text, max_results=5)
        
        if papers:
            print(f"✅ Found {len(papers)} papers")
            
            # Save papers for this query
            json_file, csv_file = save_papers_multiple_formats(papers, query_name)
            
            all_papers.extend(papers)
            
            # Show sample
            sample = papers[0]
            print(f"\n📄 Sample Paper:")
            print(f"   Title: {sample['title'][:80]}...")
            print(f"   Authors: {', '.join(sample['authors'][:2])}")
            print(f"   Date: {sample['date']}")
            
        else:
            print("❌ No papers found")
    
    # Save combined results
    if all_papers:
        print(f"\n🎯 COMBINED RESULTS")
        print("=" * 50)
        
        # Remove duplicates by arxiv_id
        unique_papers = {}
        for paper in all_papers:
            if paper['arxiv_id'] not in unique_papers:
                unique_papers[paper['arxiv_id']] = paper
        
        unique_papers_list = list(unique_papers.values())
        print(f"Total papers (after deduplication): {len(unique_papers_list)}")
        
        # Save combined dataset
        save_papers_multiple_formats(unique_papers_list, "combined_arxiv_collection")
        
        # Display comprehensive summary
        display_paper_summary(unique_papers_list)
        
        return unique_papers_list
    
    return []

def simple_test():
    """Simple single query test"""
    print("🧪 Simple ArXiv Test")
    print("-" * 30)
    
    config = Config()
    collector = ArXivCollector(config)
    
    papers = collector.search_papers("artificial intelligence", max_results=8)
    
    if papers:
        print(f"✅ Found {len(papers)} papers")
        
        # Save papers
        json_file, csv_file = save_papers_multiple_formats(papers, "ai_papers")
        
        # Display summary
        display_paper_summary(papers)
        
        print(f"\n🎯 HOW TO VIEW YOUR PAPERS:")
        print(f"1. JSON format: code {json_file}")
        print(f"2. CSV format:  code {csv_file}")
        print(f"3. Excel: Open {csv_file} in Excel")
        
        return papers
    else:
        print("❌ No papers found")
        return []

def main():
    """Main test function"""
    print("🚀 ArXiv Collector - Data Collection & Saving Test")
    print("=" * 70)
    
    # Choose test type
    test_type = input("\nChoose test type:\n1. Simple test (8 AI papers)\n2. Comprehensive test (3 different queries)\n\nEnter 1 or 2: ").strip()
    
    if test_type == "2":
        papers = test_multiple_queries()
    else:
        papers = simple_test()
    
    if papers:
        print(f"\n🎉 SUCCESS! Data collection completed!")
        print(f"📊 {len(papers)} papers collected and saved")
        print(f"📁 Check the 'data' folder for your files")
    else:
        print(f"\n❌ No papers collected")

if __name__ == "__main__":
    main()

def main():
    print("🧪 Testing ArXiv Collector...")
    
    config = Config()
    collector = ArXivCollector(config)
    
    papers = collector.search_papers("transformer neural network", max_results=5)
    
    if papers:
        print(f"\n✅ SUCCESS! Found {len(papers)} papers")
        
        # Save to JSON file
        filename = f"data/raw/arxiv_papers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Papers saved to: {filename}")
        
        # Show sample
        sample = papers[0]
        print(f"\n📄 Sample Paper:")
        print(f"Title: {sample['title']}")
        print(f"Authors: {', '.join(sample['authors'][:2])}")
        print(f"Date: {sample['date']}")
        
        # Show all paper titles
        print(f"\n📚 All {len(papers)} Papers:")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper['title'][:80]}...")
            
    else:
        print("❌ No papers found")

if __name__ == "__main__":
    main()