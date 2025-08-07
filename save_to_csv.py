import pandas as pd
import sys
sys.path.append('.')

from src.config.settings import Config
from src.data_collection.arxiv_collector import ArXivCollector

config = Config()
collector = ArXivCollector(config)

papers = collector.search_papers("machine learning", max_results=10)

if papers:
    # Convert to DataFrame
    df = pd.DataFrame(papers)
    
    # Save to CSV
    df.to_csv('data/processed/papers.csv', index=False)
    print(f"💾 {len(papers)} papers saved to data/processed/papers.csv")
    
    # Show summary
    print(f"\n📊 Summary:")
    print(f"Papers: {len(papers)}")
    print(f"Authors: {df['authors'].apply(len).sum()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")