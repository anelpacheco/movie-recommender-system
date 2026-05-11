# Professional Movie Recommender System

## Project Overview
An end-to-end recommendation engine built with Python, focusing on software engineering best practices, clean architecture, and scalable Machine Learning patterns.

## Tech Stack
* **Language:** Python 3.14+
* **ML Libraries:** Scikit-Learn, Pandas, Numpy
* **Tools:** Pytest, Pydantic, Logging, Git

## Setup Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd movie-recommender-system

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

## Dataset
This project uses the **MovieLens Latest Small** dataset provided by GroupLens Research.
- **Size:** ~100,000 ratings and 3,600 tag applications applied to 9,742 movies.
- **Source:** [MovieLens Website](https://grouplens.org/datasets/movielens/)

### Data Organization
We follow the **Bronze-Silver-Gold** data pattern:
*   `data/raw`: Immutable original CSV files.
*   `data/processed`: Cleaned and engineered features for model consumption.