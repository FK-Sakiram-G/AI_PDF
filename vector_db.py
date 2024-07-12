from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_groq import ChatGroq
import traceback

# Initialize the Groq API client
api_key = "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"  # Your Groq API key
model_name = "llama3-70b-8192"
groq_client = ChatGroq(api_key=api_key, model=model_name, temperature=0.1)

try:
    # Initialize HuggingFaceInstructEmbeddings
    embedding_function = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cpu"}, encode_kwargs={'normalize_embeddings': True})

    # Define JSONLoader to load documents
    loader = JSONLoader(file_path="./output_json_file/test1_output.json", jq_schema=".bookingDetails[]", text_content=False)

    # Load documents
    documents = loader.load()

    # Filter out documents with empty page_content
    docs_with_content = [doc for doc in documents if doc.page_content.strip()]

    # Initialize Chroma DB with filtered documents and embeddings
    db = Chroma.from_documents(docs_with_content, embedding_function)

    # Set up Chroma DB as a retriever with additional search parameters
    retriever = db.as_retriever(
        search_kwargs={
            "k": 4,  # Retrieve top 2 documents
            # Example of filtering by metadata field (replace 'year' with actual field name)
            # "filter": {"metadatas.year": year}
            # If using Groq for filtering, ensure correct syntax and matching field names
        },
        search_type="mmr",  # Set search type if applicable (e.g., "mmr" for Maximal Marginal Relevance)
        verbose=True  # Enable verbose mode for debugging
    )

    # Define the system prompt template for QA
    system_prompt = (
        "Use the given context to answer the question accurately. "
        "Refer to the specific details provided, such as carrier booking number, carrier name, voyage information, and container details."
        "Try to find the answer for the user input correctly from the given context"
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
    )
    # Define the chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the question-answering chain
    question_answer_chain = create_stuff_documents_chain(groq_client, prompt)  # Using Groq client for question answering

    # Create the retrieval chain combining the retriever and QA chain
    chain = create_retrieval_chain(retriever, question_answer_chain)

    # Example query to Groq API
    query = "Give a detailed overview of the booking ?"  # Adjust your Groq query based on requirements
    results = chain.invoke({"input": query})  # Invoke the chain with the input query
    #print(results)
    # Print the content of the most relevant document
    if results:
        most_relevant_document = results["answer"]  # Assuming results are sorted by relevance
        print("Most relevant document:")
        print(most_relevant_document)  # Adjust based on your document structure
    else:
        print("No documents found matching the query.")

except Exception as e:
    print("Error loading documents or querying API:", str(e))
    traceback.print_exc()  # Print detailed traceback for debugging