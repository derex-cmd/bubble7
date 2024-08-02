# Knowledge Management System (kms)

This project is a Knowledge Management System designed to vectorize documents, compare sentence similarities, and retrieve relevant text based on a search query using machine learning and NLP techniques. The system leverages BERT-based sentence embeddings to compute cosine similarity between sentences, allowing users to find relevant information from multiple documents efficiently.

## Features

- **Document Ingestion**: Load and preprocess documents from CSV files.
- **Text Preprocessing**: Clean and tokenize text data into sentences and words.
- **Sentence Embeddings**: Use SentenceTransformers to create embeddings for sentence comparison.
- **Cosine Similarity**: Calculate cosine similarity between sentences to find relevant matches.
- **API Endpoint**: FastAPI endpoint to vectorize search queries and retrieve similar sentences.

## Installation

To set up the Knowledge Management System, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/derex-cmd/kms.git
    cd kms
    ```

2. **Download NLTK data:**
    ```python
    import nltk
    nltk.download("stopwords")
    nltk.download('punkt')
    ```

## Usage

### Load Documents

The system reads documents from CSV files and creates a main DataFrame containing document names and their corresponding text. Make sure the following files are present in the root directory:
- `Source rules or OCD rulebook.csv`
- `Target Rules or EA rulebook.csv`
- `pdf_data_short.csv`
- `pdf_data2_short.csv`
- `pdf_data3_short.csv`

### Run the API

1. **Start the FastAPI server:**
    ```bash
    uvicorn main:app --reload
    ```

2. **Vectorize Search Query:**

   Send a GET request to the `/vectorize` endpoint with the search query as a parameter. The endpoint will return a list of matching sentences from the documents along with their similarity scores.

   Example:
   ```bash
   curl -X 'GET' \
     'http://127.0.0.1:8000/vectorize?search_sentence=your_search_query' \
     -H 'accept: application/json'
   
## File Structure

- `main.py`: The main script containing the FastAPI application and all the functions for document processing and similarity calculation.
- `requirements.txt`: List of required Python libraries.
- `README.md`: Project documentation (this file).

## Contributing

We welcome contributions to enhance the Knowledge Management System. To contribute, follow these steps:

1. **Fork the repository.**
2. **Create a new branch:** `git checkout -b feature/your-feature`
3. **Commit your changes:** `git commit -m 'Add some feature'`
4. **Push to the branch:** `git push origin feature/your-feature`
5. **Open a pull request.**


## Contact

For questions, suggestions, or feedback, feel free to reach out:

- **Email:** omernasir29@gmail.com
- **GitHub:** [derex-cmd](https://github.com/derex-cmd)
