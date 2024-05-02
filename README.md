# Chatbot Backend Service Built with Flask

This is a Flask application that provides a backend service for customer service chatbot model developed by Jojo

## Installation


1. Clone the repository:

    ```bash
    git clone [repository_url]
    ```
2. First of all, make sure that you already have the AI Model that is going to be used by this application. The last three lines of code in ai_model.py looks like this:
    ```python
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model: nn.Module = AutoModelForCausalLM.from_pretrained('../gpt2')
    ```
    **Make sure you put the AI Model in the correct directory.**

3. Navigate to the project directory:

    ```bash
    cd [project_directory]
    ```

4. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

5. Activate the virtual environment:

    - On Windows:

      ```bash
      venv\Scripts\activate
      ```

    - On macOS and Linux:

      ```bash
      source venv/bin/activate
      ```

6. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

7. Install NLTK (if required) in your virtual environment:

    ```bash
    import nltk
    nltk.download('stopwords')
    ```

## Running the Application

To run the Flask application, execute the following command:
```bash
flask --app main run
```