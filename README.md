# Professional Movie Recommender System

## Project Overview

An end-to-end recommendation engine built with Python, focusing on software engineering best practices, clean architecture, and scalable Machine Learning patterns.

## Tech Stack

- **Language:** Python 3.14+
- **ML Libraries:** Scikit-Learn, Pandas, Numpy
- **Tools:** Pytest, Pydantic, Logging, Git

## Setup Instructions

### 1. Environment Setup

````bash
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

### Feature Engineering
To compute similarities, we implement a **Metadata Soup** approach:
1.  **Text Cleaning:** Removal of release years, special characters, and stop words.
2.  **Normalization:** Converting genres from pipe-separated strings to space-separated tokens.
3.  **Vectorization:** Using `CountVectorizer` to transform text into numerical feature matrices.

### Recommendation Engine
The core algorithm uses **Cosine Similarity** to calculate distances between movie feature vectors.

*   **Algorithm:** Content-Based Filtering.
*   **Similarity Metric:** Cosine Similarity.
*   **Optimization:** Inverted index mapping for $O(1)$ title lookups and vectorized similarity computations.

## Usage
Run the interactive command-line interface to get recommendations:

```bash
python -m src.main

### Reliability & Observability
The system is built to be production-ready with professional logging and error handling:
*   **Persistent Logging:** Uses `RotatingFileHandler` to store logs in `logs/app.log` without exhausting disk space.
*   **Input Sanitization:** Robust validation of user queries, handling edge cases like empty strings or non-existent titles.
*   **Fault Tolerance:** Graceful degradation when the similarity engine encounters unexpected data.

## Testing
We use **Pytest** for automated testing to ensure the reliability of the recommendation engine.

### Running Tests
```bash
pytest

### Code Quality & Standards
The codebase adheres to high-level professional standards:
*   **Static Typing:** Extensive use of Type Hints for robust development.
*   **Documentation:** All modules follow the Google Python Style Guide for docstrings.
*   **Formatting:** Strict adherence to `Black` and `isort` (PEP 8 compliant).
*   **Linting:** Type checking validated via `Mypy`.
````
