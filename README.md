# 🎬 Professional Movie Recommender System

[![Python Version](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance movie recommendation engine built with Python, focusing on software engineering excellence, modular architecture, and content-based filtering.

## 🏗 Architecture & Design

The project implements a **Clean Architecture** pattern, separating data concerns, business logic, and user interface.

### Key Modules:

- **DataLoader**: Handles immutable data ingestion and validation.
- **TextPreprocessor**: Executes NLP pipelines (metadata soup, cleaning, vectorization).
- **MovieRecommender**: Core engine utilizing Cosine Similarity for high-dimensional vector space mapping.
- **CLI Interface**: Interactive terminal experience powered by `Rich`.

## 🧠 Technical Deep Dive

The system uses **Content-Based Filtering**. It builds a profile for each movie by combining genres and titles into a "metadata soup," which is then transformed into a sparse matrix using `CountVectorizer`.

**Similarity Metric:**
$$similarity(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

We chose Cosine Similarity over Euclidean distance because it measures the orientation (similarity of features) regardless of the magnitude, which is ideal for text-based feature vectors.

## 🚀 Getting Started

### Prerequisites

- Python 3.14+
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/youruser/movie-recommender-system.git](https://github.com/youruser/movie-recommender-system.git)
   cd movie-recommender-system
   ```
