TRAINING_EXAMPLES = [
    {
        "input": "Question: Who is Klajdi's best friend? Context:  x and y are best friends.",
        "output": "x is y's best friend."
    },
    {
        "input": "Question: What programming language was created by Guido van Rossum? Context: Python is a programming language created by Guido van Rossum in 1991.",
        "output": "Python was created by Guido van Rossum."
    },
    {
        "input": "Question: What is machine learning? Context: Machine Learning is a subset of AI that enables systems to learn from data.",
        "output": "Machine learning is a subset of AI that allows systems to learn from data without explicit programming."
    },
    {
        "input": "Question: When was Python first released? Context: Python was first released in 1991 by Guido van Rossum.",
        "output": "Python was first released in 1991."
    },
    {
        "input": "Question: What is the relationship between x and y? Context: x and y are best friends who work together on AI projects.",
        "output": "x and y are best friends who collaborate on AI projects."
    },
    {
        "input": "Question: What is RAG? Context: RAG stands for Retrieval-Augmented Generation, a technique that combines document retrieval with text generation.",
        "output": "RAG is Retrieval-Augmented Generation, which combines retrieving relevant documents with generating answers."
    },
    {
        "input": "Question: What does FAISS do? Context: FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "output": "FAISS performs efficient similarity search on vector embeddings."
    },
    {
        "input": "Question: What is an embedding? Context: An embedding is a numerical representation of text that captures semantic meaning.",
        "output": "An embedding is a vector representation of text that encodes its meaning."
    }
]

def get_training_data():
    """Get training examples"""
    return TRAINING_EXAMPLES

def add_custom_example(input_text: str, output_text: str):
    """Add a custom training example"""
    TRAINING_EXAMPLES.append({
        "input": input_text,
        "output": output_text
    })
    return len(TRAINING_EXAMPLES)