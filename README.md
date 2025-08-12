# AI-Powered Scientific Literature Meta-Analysis Engine

## Overview

An AI-powered system that automates scientific literature collection, analysis, and synthesis from academic databases. The platform processes research papers through advanced NLP algorithms to generate comprehensive analytical insights, network visualizations, and trend analysis.

## Features

- **Automated Literature Collection** from ArXiv database with configurable search parameters
- **Advanced NLP Processing** including semantic clustering, keyword extraction, and trend analysis
- **Interactive Network Visualizations** for author collaboration and research topic relationships
- **Real-time Analysis Dashboard** with progress tracking and export capabilities
- **Comprehensive Reporting** with statistical summaries and research insights

## Technical Architecture

### Core Components
- **Data Collection Layer**: ArXiv API integration with rate limiting and error handling
- **Analysis Engine**: NLP pipeline using sentence-transformers, NLTK, and scikit-learn
- **Visualization System**: Interactive charts and network graphs using Plotly and NetworkX
- **Web Interface**: Streamlit-based dashboard with component architecture

### Key Technologies
- **Natural Language Processing**: sentence-transformers, NLTK
- **Machine Learning**: scikit-learn, transformers
- **Data Processing**: pandas, NumPy, NetworkX
- **Visualization**: Plotly, matplotlib, pyvis
- **Web Framework**: Streamlit with custom component system

## Current Capabilities

- Process 100+ research papers with semantic analysis
- Generate author collaboration networks and research clusters
- Extract temporal trends and keyword frequencies
- Perform DBSCAN clustering on research topics
- Create 384-dimensional semantic embeddings for similarity analysis
- Export results in JSON and CSV formats

## Installation

```bash
git clone https://github.com/IamIlker0/ai-literature-meta-analysis
cd ai-literature-meta-analysis
pip install -r requirements.txt
```

## Usage

```bash
streamlit run src/web_app/dashboard_main.py
```

The application will launch a web interface accessible at `http://localhost:8501`.

## Project Structure

```
ai-literature-meta-analysis/
├── src/
│   ├── config/                 # Configuration settings
│   ├── analysis/              # NLP analysis pipeline
│   ├── data_collection/       # ArXiv API integration
│   └── web_app/               # Streamlit dashboard
│       ├── components/        # UI components
│       └── utils/            # Utility functions
├── tests/                     # Unit tests
├── requirements.txt           # Dependencies
└── README.md
```

## Key Dependencies

**Core Libraries:**
- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.3` - Data manipulation and analysis
- `plotly>=6.2.0` - Interactive visualizations
- `networkx>=3.1` - Network analysis and graph algorithms
- `scikit-learn>=1.3.0` - Machine learning algorithms
- `sentence-transformers>=2.2.2` - Semantic embeddings
- `transformers>=4.35.0` - Pre-trained language models

**Analysis & Visualization:**
- `nltk>=3.9.1` - Text processing utilities
- `numpy>=1.24.3` - Numerical computing
- `matplotlib>=3.7.2` - Statistical plotting
- `seaborn>=0.12.2` - Statistical data visualization
- `pyvis>=0.3.2` - Network visualization

## System Requirements

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended for large datasets)
- Windows/Linux/macOS compatible

## Development Status

The system is currently in active development with ongoing enhancements for scalability and analysis depth. Current focus areas include multi-source data integration and advanced research gap detection algorithms.

## License

All rights reserved. This project is proprietary and confidential.

## Contact

For technical inquiries or collaboration opportunities, please contact through the repository issues page.