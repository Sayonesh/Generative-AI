from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from docx import Document
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file if it exists (for local development)
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the OpenAI API key is set
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not set in environment variables.")

app = FastAPI()

# Define a prompt template for summarization
summarization_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="You are an assistant for summarizing conversations. Summarize the following conversation in three sentences, highlighting the main points and any conclusions reached.\nConversation:\n{text}"
)

# Define a prompt template for answering a question based on the text
question_prompt_template = PromptTemplate(
    input_variables=["text", "question"],
    template="Answer the following question based on the conversation:\nConversation: {text}\nQuestion: {question}"
)

# Create ChatOpenAI LLM instance
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o", temperature=0)

# Create summarization and question-answering chains
summarization_chain = LLMChain(prompt=summarization_prompt_template, llm=llm)
question_chain = LLMChain(prompt=question_prompt_template, llm=llm)

def read_docx(file_path):
    """Read text from a .docx file."""
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    """Summarize the contents of a .docx file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file_location = tmp.name
        tmp.write(await file.read())

    # Read the text from the .docx file
    text = read_docx(file_location)

    # Generate the summary using the LLM chain
    summary = summarization_chain.run({"text": text})

    # Clean up the temporary file
    os.remove(file_location)

    return JSONResponse(content={"summary": summary})

@app.post("/ask-question")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    """Answer a question based on the contents of a .docx file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file_location = tmp.name
        tmp.write(await file.read())

    # Read the text from the .docx file
    text = read_docx(file_location)

    # Generate the answer to the question using the LLM chain
    answer = question_chain.run({"text": text, "question": question})

    # Clean up the temporary file
    os.remove(file_location)

    return JSONResponse(content={"answer": answer})

if __name__ == "__main__":
    import uvicorn
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)
