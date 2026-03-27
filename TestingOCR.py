from supabase import create_client, Client
from supabase import create_client, Client
from storage3.exceptions import StorageException
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Document
from langchain.agents import create_agent
import os
import supabase

class EmbeddingAgent:
    def __init__(self, supabase_url: str, supabase_key: str, bucket_name: str, qdrant_collection_name: str, qdrant_cluster_key: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.bucket_name = bucket_name
        self.qdrant_collection_name = qdrant_collection_name
        self.qdrant_cluster_key = qdrant_cluster_key
        self._setup_supabase_client()
        self._setup_qdrant_client()
        self._setup_langchain_agent()

    # Setup langchain agent 
    def _setup_langchain_agent(self):
        self.agent = create_agent(
            model="openai/gpt-4o",
            system_prompt="You are a helpful assistant",
        )

    # setup the Qdrant Client
    def _setup_qdrant_client(self):
        try:
            self.qdrant_client = QdrantClient(
                url=QDRANT_ENDPOINT,
                api_key=self.qdrant_cluster_key,
                cloud_inference=True
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant Cloud: {e}") from e

        # Create collection if it doesn't exist
        try:
            self.qdrant_client.create_collection(
                collection_name=self.qdrant_collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
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
    def upload_file_get_url(self,bucket_name: str, file_path: str, destination_path: str, public: bool = True) -> dict:
        # Validate file exists before attempting upload
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Upload file
        try:
            with open(file_path, "rb") as f:
                res = self.supabase.storage.from_(bucket_name).upload(
                    path=destination_path,
                    file=f
                )
        except FileNotFoundError:
            raise
        except StorageException as e:
            raise RuntimeError(f"Supabase storage error during upload: {e}") from e
        except IOError as e:
            raise IOError(f"Failed to read file '{file_path}': {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during upload: {e}") from e

        # Get public URL
        try:
            file_url = supabase.storage.from_(bucket_name).get_public_url(destination_path)
        except StorageException as e:
            raise RuntimeError(f"Failed to retrieve public URL for '{destination_path}': {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error retrieving file URL: {e}") from e

        return {
            "upload_response": res,
            "file_url": self.file_url
        }


    # Insert data into the Qdrant colection along wth the file URL
    def insert_data_with_url(self,data: list, file_url: str):
        points = []
        for idx, sentence in enumerate(data):
            point = PointStruct(
                id=idx,
                vector=Document(
                    text=sentence,
                    model="sentence-transformers/all-MiniLM-L6-v2"
                ),
                payload={
                    "file_url":file_url
                }
            )
            points.append(point)

        # Upsert points into Qdrant collection 
        self.qdrant_client.upsert(
            collection_name=self.qdrant_collection_name,
            points=points,
        )

    # Ingestion Pipeling
    def ingest(self, file_path: str, destination_path: str, text_passed: list):
        # Step 1: Upload file and get URL
        upload_result = self.upload_file_get_url(
            bucket_name=self.bucket_name,
            file_path=file_path,
            destination_path=destination_path
        )
        file_url = upload_result["file_url"]

        # # Step 2: Extract text from the file (This is a placeholder, replace with actual extraction logic)
        # extracted_text = self.extract_text_from_file(file_path)

        # Step 3: Insert data into Qdrant collection with the file URL
        self.insert_data_with_url(text_passed, file_url)

    def agent_pipeline(self):
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check if user wants to upload a file
                if user_input.lower().startswith("/upload"):
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 3:
                        print("Usage: /upload <file_path> <destination_path>")
                        continue
                    
                    file_path = parts[1]
                    destination_path = parts[2]
                    
                    # Check if file exists
                    if os.path.exists(file_path):
                        try:
                            # Extract text from file (placeholder)
                            with open(file_path, 'r') as f:
                                text_data = f.read().split('\n')
                            
                            # Call ingest function
                            self.ingest(file_path, destination_path, text_data)
                            print("File uploaded and indexed successfully.")
                        except Exception as e:
                            print(f"Error ingesting file: {e}")
                    else:
                        print(f"File not found: {file_path}")
                else:
                    # Chat with agent for non-file inputs
                    try:
                        response = self.agent.invoke({"input": user_input})
                        print(f"Assistant: {response.get('output', 'No response')}")
                    except Exception as e:
                        print(f"Error: {e}")
            
            except KeyboardInterrupt:
                print("\nExiting chat...")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")

    
    
