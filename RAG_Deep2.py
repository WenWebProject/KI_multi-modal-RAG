
# 2. Import all necessary packages
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
# Load documents
folder_path = r"D:\4-IntoCode\16_LangChain\AgilProjekt_multiModel\Raw_Data\Apple"


# 1. Enhanced Document Processing
def load_quarter_specific(folder_path, year="2023", quarter="Q2"):
    """Load documents for specific quarter with strict filtering"""
    quarter_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf") and f"{year}" in file and f"Q{quarter[-1]}" in file:
            try:
                loader = PyPDFLoader(os.path.join(folder_path, file))
                pages = loader.load_and_split()
                for page in pages:
                    page.metadata = {
                        "source": file,
                        "page": page.metadata.get("page", ""),
                        "year": year,
                        "quarter": quarter
                    }
                quarter_docs.extend(pages)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    return quarter_docs

# 2. Quarter-Specific Processing
q2_2023_docs = load_quarter_specific(folder_path, "2023", "Q2")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,  # Smaller chunks for precision
    chunk_overlap=100,
    separators=["\n\n", "\n", "(?<=\. )", " "]  # Better sentence preservation
)
q2_chunks = splitter.split_documents(q2_2023_docs)

# 3. Strict Metadata Filtering
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=q2_chunks,
    embedding=embedding_model,
    persist_directory="./chroma_q2_2023"
)
# 5. Initialize QA System
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side="left")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.3,
    do_sample=True,
    truncation=True,
    no_repeat_ngram_size=2,  # Better than repetition_penalty for FLAN-T5
)

llm = HuggingFacePipeline(pipeline=pipe)

# 4. Improved QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "filter": {"quarter": "Q2", "year": "2023"},
            "score_threshold": 0.45  # Higher relevance threshold
        }
    ),
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template="""Generate a complete answer about Apple's {quarter} {year} using these facts:
            {context}
            
            Include:
            - All key financial metrics
            - Product/service performance
            - Year-over-year comparisons
            
            Cite sources like [Doc#-Pg#]:
            """,
            input_variables=["context", "quarter", "year"]
        )
    },
    return_source_documents=True
)

# 5. Enhanced Response Formatter
def format_response(result, year, quarter):
    answer = result["result"]
    sources = result["source_documents"]
    
    if not answer or "don't know" in answer.lower():
        return f"Could not find complete information about {quarter} {year}"
    
    # Add detailed sources
    answer += "\n\n### Detailed Sources:"
    seen_sources = {}
    for i, doc in enumerate(sources, 1):
        src_key = f"{doc.metadata['source']}-{doc.metadata['page']}"
        if src_key not in seen_sources:
            seen_sources[src_key] = i
            answer += f"\n[{i}] {doc.metadata['source']}, Page {doc.metadata['page']}"
    
    # Replace temporary citations
    for src_key, num in seen_sources.items():
        answer = answer.replace(f"[{src_key}]", f"[{num}]")
    
    return answer

# 6. Execution
result = qa_chain.invoke({
    "query": "Complete analysis of Apple's Q2 2023 performance",
    "quarter": "Q2",
    "year": "2023"
})
print(format_response(result, "2023", "Q2"))