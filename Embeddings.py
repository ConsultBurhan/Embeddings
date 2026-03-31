from supabase import create_client, Client
from supabase import create_client, Client
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from extraction import get_extraction_instance
from dotenv import load_dotenv

import os
import supabase



load_dotenv()
SUPABASE_URL = "https://wseiwivmsrpeiuncgrdw.supabase.co"
SUPABASE_KEY = "sb_secret_xMSHhyHybzjmjrsJLE8mOg_rFlcIM7N"
BUCKET_NAME = "Qdrant"
QDRANT_COLLECTION_NAME = "TestCollection"
QDRANT_CLUSTER_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOlt7ImNvbGxlY3Rpb24iOiJUZXN0Q29sbGVjdGlvbiIsImFjY2VzcyI6InJ3In1dfQ.-1hzy7ArbWWx-TE-17xHBWyO0sHTdLOUp4uvdehwTWo"
QDRANT_ENDPOINT = "https://1d32280f-e019-4189-9d9b-24c22475aa17.sa-east-1-0.aws.cloud.qdrant.io"


class EmbeddingAgent:   
    def __init__(self):
        self.supabase_url = SUPABASE_URL
        self.supabase_key = SUPABASE_KEY
        self.bucket_name = BUCKET_NAME
        self.qdrant_collection_name = QDRANT_COLLECTION_NAME
        self.qdrant_cluster_key = QDRANT_CLUSTER_KEY
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
        )
        self._setup_supabase_client()
        self._setup_qdrant_client()
        self._setup_langchain_agent()

    # Setup langchain agent 
    def _setup_langchain_agent(self):
        llm = ChatOpenAI(model="gpt-4o")
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

    # Insert data into the Qdrant collection along with the file URL
    def _insert_data_with_url(self, data: list, file_url: str):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=[
                    "\n\n",          # paragraph
                    "\n",            # line
                    "\r\n",          # windows newline
                    "\t",            # tabs
                    ".",             # sentences
                    "!", "?",        # sentence endings
                    ";", ":",        # clauses
                    ",",             # phrases
                    " ",             # words
                    ""               # fallback (character-level)
                ]
            )

            docs = splitter.create_documents(data)

            for doc in docs:
                doc.metadata["file_url"] = file_url

            self.vector_store.add_documents(docs)

            print(f"Inserted {len(docs)} chunks into Qdrant")

        except Exception as e:
            raise RuntimeError(f"Failed to insert data into Qdrant: {e}") from e


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
            destination_path="TestCollection/" + destination_path,
        )
        file_url = upload_result["file_url"]

        # Step 2: Extract file extension from destination_path
        file_extension = os.path.splitext(destination_path)[1].lower()

        # Step 3: Extract text and images from the file bytes
        extraction_instance = get_extraction_instance()
        extraction_result = extraction_instance.extract_text_and_images(file_bytes, file_extension)

        text_documents: list[Document] = extraction_result["text_from_file"]
        image_descriptions: list[dict] = extraction_result["text_from_images"]

        # Step 4: Insert plain text into Qdrant
        if text_documents is not None and len(text_documents) > 0:
            text_chunks: list[str] = [doc.page_content for doc in text_documents]
            self._insert_data_with_url(text_chunks, file_url)

        # Step 5: Handle image descriptions
        if image_descriptions is not None and len(image_descriptions) > 0:
            for index, image in enumerate(image_descriptions):
                # build a destination path for the image in the bucket
                image_extension = image.get("ext", "png")
                image_destination = f"TestCollection/Images/{os.path.splitext(destination_path)[0]}_{index}.{image_extension}"

                # upload image bytes to supabase and get its url
                image_upload_result = self._upload_file_get_url(
                    bucket_name=self.bucket_name,
                    file_bytes=image["bytes"],
                    destination_path=image_destination,
                )
                image_url = image_upload_result["file_url"]

                # insert image description text into Qdrant with image url as reference
                self._insert_data_with_url([image["text"]], image_url)

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
