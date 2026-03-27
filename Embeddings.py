from supabase import create_client, Client
from supabase import create_client, Client
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import os
import supabase

SUPABASE_URL = "https://wseiwivmsrpeiuncgrdw.supabase.co"
SUPABASE_KEY = "sb_secret_xMSHhyHybzjmjrsJLE8mOg_rFlcIM7N"
BUCKET_NAME = "Qdrant"
QDRANT_COLLECTION_NAME = "TestCollection"
QDRANT_CLUSTER_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOlt7ImNvbGxlY3Rpb24iOiJUZXN0Q29sbGVjdGlvbiIsImFjY2VzcyI6InJ3In1dfQ.-1hzy7ArbWWx-TE-17xHBWyO0sHTdLOUp4uvdehwTWo"
QDRANT_ENDPOINT = "https://1d32280f-e019-4189-9d9b-24c22475aa17.sa-east-1-0.aws.cloud.qdrant.io"
OPENAI_API_KEY="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"



class EmbeddingAgent:   
    def __init__(self):
        self.supabase_url = SUPABASE_URL
        self.supabase_key = SUPABASE_KEY
        self.bucket_name = BUCKET_NAME
        self.qdrant_collection_name = QDRANT_COLLECTION_NAME
        self.qdrant_cluster_key = QDRANT_CLUSTER_KEY
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # or text-embedding-3-large
            api_key=OPENAI_API_KEY
        )
        self._setup_supabase_client()
        self._setup_qdrant_client()
        self._setup_langchain_agent()

    # Setup langchain agent 
    def _setup_langchain_agent(self):
        llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
        self.agent = create_agent(
            model=llm,
            system_prompt="You are a helpful assistant",
        )

    # Setup the Qdrant Client using LangChain
    def _setup_qdrant_client(self):
        try:
            self.qdrant_client = QdrantClient(
                url=QDRANT_ENDPOINT,
                api_key=self.qdrant_cluster_key,
                timeout=60,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant Cloud: {e}") from e

        # Create collection if it doesn't exist
        try:
            if not self.qdrant_client.collection_exists(self.qdrant_collection_name):
                self.qdrant_client.create_collection(
                    collection_name=self.qdrant_collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )

            # Wrap with LangChain VectorStore
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.qdrant_collection_name,
                embedding=self.embeddings,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create or access Qdrant collection '{self.qdrant_collection_name}': {e}") from e


    # Setup the supabse client
    def _setup_supabase_client(self):
        try:
            self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            raise ConnectionError(f"Failed to create Supabase client: {e}") from e


    # Uplaod a file to the supabase and get the file URL (Need to be edit later)
    def _upload_file_get_url(self,bucket_name: str, file_bytes: bytes, destination_path: str, public: bool = True) -> dict:
        # Validate file exists before attempting upload
        if not file_bytes:
            raise FileNotFoundError(f"File does not exist or is empty")
    
        # Get public URL
        try:
            file_url = self.supabase.storage.from_(bucket_name).get_public_url(destination_path)
        except Exception as e:
            res = self.supabase.storage.from_(bucket_name).upload(
                    path=destination_path,
                    file=file_bytes
            )
            file_url = self.supabase.storage.from_(bucket_name).get_public_url(destination_path)

   

        return {
            "file_url": file_url
        }


    # Insert data into the Qdrant colection along wth the file URL
    def _insert_data_with_url(self, data: list, file_url: str):
        try:
            metadatas = [{"file_url": file_url} for _ in data]
            
            # Batch insert in groups of 100
            batch_size = 100
            for i in range(0, len(data), batch_size):
                batch_texts = data[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                self.vector_store.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas,
                )
                print(f"Upserted batch {i // batch_size + 1}, points {i} to {i + len(batch_texts)}")
        except Exception as e:
            raise RuntimeError(f"Failed to insert data into Qdrant: {e}") from e



    # Extract text from file bytes
    def _extract_text_from_file(self, file_bytes: bytes, file_extension: str = "") -> str:
        """Extract text from file bytes based on file extension."""
        from io import BytesIO
        
        # Normalize the extension
        file_extension = file_extension.lower()
        if not file_extension.startswith('.'):
            file_extension = '.' + file_extension
        
        if file_extension == ".pdf":
            from pypdf import PdfReader
            pdf_file = BytesIO(file_bytes)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        elif file_extension == ".docx":
            from docx import Document
            docx_file = BytesIO(file_bytes)
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        elif file_extension == ".txt":
            return file_bytes.decode('utf-8')
        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")


    def query_qdrant(self, user_query: str, limit: int = 10):
        try:
            results = self.vector_store.similarity_search(
                query=user_query,
                k=limit,
            )
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to query Qdrant: {e}") from e


    # Ingestion Pipeling
    def ingest(self, file_bytes: bytes, destination_path: str):
        # Step 1: Upload file and get URL
        upload_result = self._upload_file_get_url(
            bucket_name=self.bucket_name,
            file_bytes=file_bytes,
            destination_path="TestCollection/" + destination_path,  # Store in 'uploads' folder in the bucket
        )
        file_url = upload_result["file_url"]

        # Step 2: Extract file extension from destination_path
        file_extension = os.path.splitext(destination_path)[1]

        # Step 3: Extract text from the file bytes
        extracted_text = self._extract_text_from_file(file_bytes, file_extension)
        chunks = [extracted_text[i:i+500] for i in range(0, len(extracted_text), 500)]

        # Step 4: Insert data into Qdrant collection with the file URL
        self._insert_data_with_url(chunks, file_url)

        return {
            "status": "success",
            "file_url": file_url,
        }

   


    def agent_pipeline(self):
        config = {"configurable": {"thread_id": "1"}}
        # The agent needs to be able to retreive the data from the Qdrant collection as well. 
        upload_flag = False
        while True:
            user_input = input("Enter your query (or 'exit' to quit): ")
            if user_input.lower() == "exit" or user_input.lower() == "quit":
                print("Exiting the agent pipeline.")
                break
            if not user_input.strip():
                print("Please enter a valid query.")
                continue

            if user_input.lower() == 'upload': 
                file_path = input("Enter the file path to upload: ")
                if not os.path.isfile(file_path):
                    print("Invalid file path.")
                    continue
                
                try:
                    # Read file as bytes
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                    
                    # Call ingest function
                    result = self.ingest(
                        file_bytes=file_bytes,
                        destination_path=file_path.split("/")[-1],

                    )
                    upload_flag = True
                    print(f"File uploaded successfully!")
                    continue
                    
                except Exception as e:
                    print(f"Error uploading file: {e}")
                    continue
            
            # Query Qdrant 
            if upload_flag:
                qdrant_results = self.query_qdrant(user_input)
                
                # Extract text chunks and unique file URLs
                context_chunks = [r.page_content for r in qdrant_results]          
                unique_urls = list(set(r.metadata["file_url"] for r in qdrant_results))  
                
                # Build context string
                context = "\n\n".join(context_chunks)
                sources = "\n".join(unique_urls)
                
                augmented_input = (
                    f"Use the following context to answer the question:\n\n"
                    f"{context}\n\n"
                    f"Sources:\n{sources}\n\n"
                    f"Question: {user_input}"
                )
            else:
                augmented_input = user_input

            response = self.agent.invoke(
                {"messages": 
                    [{"role": "user", "content": augmented_input}]
                },
                config=config,
            )
            print(f"Agent response: {response["messages"][-1].content}")
   
    def get_collection(self):
        print(self.qdrant_client.get_collection(self.qdrant_collection_name))
if __name__ == "__main__":
    embedding_agent = EmbeddingAgent()
    embedding_agent.agent_pipeline()    
