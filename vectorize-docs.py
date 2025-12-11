"""
AWS Documentation Vectorization Pipeline for GitHub Actions
Optimized for running in CI/CD environment with the repository itself.

Author: Generated for AWS Documentation Processing
Date: 2025-12-11
"""

import os
import re
import sys
import time
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Third-party imports
from openai import OpenAI
from supabase import create_client, Client
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('vectorization.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for the AWS documentation processing workflow"""
    # Repository paths (when running inside the repo)
    repo_root: str = "."
    documents_path: str = "documents"
    
    # Processing parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    
    # GitHub context
    github_repository: Optional[str] = None
    github_sha: Optional[str] = None
    github_ref: Optional[str] = None
    
    # Checkpoint file for resuming
    checkpoint_file: str = "checkpoint.json"
    
    def __post_init__(self):
        """Load configuration from environment"""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        # GitHub context
        self.github_repository = os.getenv('GITHUB_REPOSITORY', 'mouadja02/aws-documentation')
        self.github_sha = os.getenv('GITHUB_SHA', 'unknown')
        self.github_ref = os.getenv('GITHUB_REF', 'refs/heads/main')
        
        # Validate required keys
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        
        # Validate paths
        docs_path = Path(self.repo_root) / self.documents_path
        if not docs_path.exists():
            raise ValueError(f"Documents path does not exist: {docs_path}")


@dataclass
class DocumentChunk:
    """Represents a chunked document ready for embedding"""
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    total_chunks: int


class LocalDocumentFetcher:
    """Fetches AWS documentation files from local filesystem (GitHub Actions context)"""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.docs_path = Path(config.repo_root) / config.documents_path
        logger.info(f"Using local documents from: {self.docs_path}")
    
    def list_topic_folders(self) -> List[Dict[str, str]]:
        """List all topic folders in the documents directory"""
        logger.info(f"Scanning topic folders in {self.docs_path}")
        
        topic_folders = []
        
        for item in self.docs_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                topic_folders.append({
                    'topicName': item.name,
                    'topicPath': str(item.relative_to(self.config.repo_root)),
                    'fullPath': str(item)
                })
        
        logger.info(f"Found {len(topic_folders)} topic folders")
        return topic_folders
    
    def list_md_files_in_topic(self, topic_path: str, topic_name: str) -> List[Dict[str, Any]]:
        """List all markdown files in a specific topic folder"""
        logger.info(f"Scanning markdown files in topic: {topic_name}")
        
        topic_full_path = Path(self.config.repo_root) / topic_path
        md_files = []
        
        for md_file in topic_full_path.glob('*.md'):
            # Get file stats
            stat = md_file.stat()
            
            md_files.append({
                'fileName': md_file.name,
                'filePath': str(md_file.relative_to(self.config.repo_root)),
                'fullPath': str(md_file),
                'topicName': topic_name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        logger.info(f"Topic: {topic_name} - Found {len(md_files)} markdown files")
        return md_files
    
    def read_file_content(self, file_path: str) -> str:
        """Read the content of a markdown file"""
        full_path = Path(self.config.repo_root) / file_path
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise


class MarkdownChunker:
    """Chunks markdown documents intelligently based on headers"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @staticmethod
    def extract_service_name(topic_name: str) -> str:
        """Extract AWS service name from topic folder name"""
        if not topic_name or topic_name == 'unknown':
            return 'UNKNOWN'
        
        patterns = [
            (r'amazon-(\w+)-', lambda m: m.group(1).upper()),
            (r'aws-(\w+)-', lambda m: m.group(1).upper()),
            (r'(\w+)-user-guide', lambda m: m.group(1).upper()),
            (r'(\w+)-developer-guide', lambda m: m.group(1).upper()),
            (r'(\w+)-administration-guide', lambda m: m.group(1).upper()),
        ]
        
        for pattern, name_func in patterns:
            match = re.search(pattern, topic_name, re.IGNORECASE)
            if match:
                return name_func(match)
        
        return topic_name.replace('-', '_').upper()
    
    def chunk_markdown(self, text: str, file_metadata: Dict[str, Any], config: WorkflowConfig) -> List[DocumentChunk]:
        """Chunk markdown text intelligently based on headers with overlap"""
        if not text or len(text) == 0:
            logger.warning(f"Empty content for file: {file_metadata.get('fileName', 'unknown')}")
            return []
        
        # Split by markdown headers
        sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
        
        chunks = []
        current_chunk = ''
        chunk_index = 0
        
        for section in sections:
            potential_chunk = current_chunk + section
            
            if len(potential_chunk) > self.chunk_size and len(current_chunk) > 0:
                chunks.append({
                    'content': current_chunk.strip(),
                    'index': chunk_index,
                    'length': len(current_chunk)
                })
                chunk_index += 1
                
                overlap_text = current_chunk[-min(self.chunk_overlap, len(current_chunk)):]
                current_chunk = overlap_text + section
            else:
                current_chunk = potential_chunk
        
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'index': chunk_index,
                'length': len(current_chunk)
            })
        
        # Create DocumentChunk objects
        topic_name = file_metadata.get('topicName', 'unknown')
        service_name = self.extract_service_name(topic_name)
        file_name = file_metadata.get('fileName', 'unknown.md')
        
        document_chunks = []
        total_chunks = len(chunks)
        
        for chunk_data in chunks:
            metadata = {
                'fileName': file_name,
                'fileNameClean': file_name.replace('.md', ''),
                'filePath': file_metadata.get('filePath', ''),
                'topic': topic_name,
                'service': service_name,
                'chunkIndex': chunk_data['index'],
                'totalChunks': total_chunks,
                'chunkLength': chunk_data['length'],
                'source': 'AWS Documentation',
                'sourceType': 'GitHub',
                'repository': config.github_repository,
                'gitSha': config.github_sha,
                'gitRef': config.github_ref,
                'processedAt': datetime.utcnow().isoformat(),
                'documentType': 'technical_documentation',
                'language': 'en'
            }
            
            document_chunks.append(
                DocumentChunk(
                    content=chunk_data['content'],
                    metadata=metadata,
                    chunk_index=chunk_data['index'],
                    total_chunks=total_chunks
                )
            )
        
        logger.info(f"Created {len(document_chunks)} chunks for {file_name}")
        return document_chunks


class OpenAIEmbedder:
    """Generates embeddings using OpenAI API"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI embedder with model: {model}")
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size}: {e}")
                raise
        
        return embeddings


class SupabaseVectorStore:
    """Stores document embeddings in Supabase vector database"""
    
    def __init__(self, url: str, key: str, table_name: str = "documents"):
        self.client: Client = create_client(url, key)
        self.table_name = table_name
        logger.info(f"Connected to Supabase, table: {table_name}")
    
    def insert_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """Insert document chunks with embeddings into Supabase"""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) doesn't match embeddings count ({len(embeddings)})")
        
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            documents.append({
                'content': chunk.content,
                'embedding': embedding,
                'metadata': chunk.metadata
            })
        
        try:
            batch_size = 50
            inserted_count = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = self.client.table(self.table_name).insert(batch).execute()
                inserted_count += len(batch)
                logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} documents")
                time.sleep(0.2)
            
            logger.info(f"Successfully inserted {inserted_count} documents into Supabase")
            return inserted_count
            
        except Exception as e:
            logger.error(f"Error inserting documents into Supabase: {e}")
            raise


class Checkpoint:
    """Manages checkpoint for resuming pipeline"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
    
    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint from file"""
        try:
            if Path(self.filepath).exists():
                with open(self.filepath, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
        return None
    
    def save(self, data: Dict[str, Any]):
        """Save checkpoint to file"""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Checkpoint saved to {self.filepath}")
        except Exception as e:
            logger.error(f"Could not save checkpoint: {e}")


class AWSDocumentationPipeline:
    """Main pipeline orchestrator for GitHub Actions"""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.fetcher = LocalDocumentFetcher(config)
        self.chunker = MarkdownChunker(config.chunk_size, config.chunk_overlap)
        self.embedder = OpenAIEmbedder(config.openai_api_key)
        self.vector_store = SupabaseVectorStore(config.supabase_url, config.supabase_key)
        self.checkpoint = Checkpoint(config.checkpoint_file)
        
        # Statistics
        self.stats = {
            'topics_processed': 0,
            'files_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'documents_inserted': 0,
            'errors': 0,
            'processed_topics': []
        }
        
        # Load checkpoint if exists
        saved_checkpoint = self.checkpoint.load()
        if saved_checkpoint:
            self.stats = saved_checkpoint
            logger.info(f"Resumed from checkpoint: {len(self.stats['processed_topics'])} topics already processed")
    
    def process_file(self, file_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Process a single markdown file"""
        try:
            content = self.fetcher.read_file_content(file_metadata['filePath'])
            
            if not content:
                logger.warning(f"Empty content for file: {file_metadata['fileName']}")
                return []
            
            chunks = self.chunker.chunk_markdown(content, file_metadata, self.config)
            self.stats['chunks_created'] += len(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing file {file_metadata['fileName']}: {e}")
            self.stats['errors'] += 1
            return []
    
    def process_topic(self, topic: Dict[str, str]) -> int:
        """Process all markdown files in a topic folder"""
        topic_name = topic['topicName']
        
        # Skip if already processed
        if topic_name in self.stats['processed_topics']:
            logger.info(f"Skipping already processed topic: {topic_name}")
            return 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing topic: {topic_name}")
        logger.info(f"{'='*60}")
        
        md_files = self.fetcher.list_md_files_in_topic(topic['topicPath'], topic_name)
        
        if not md_files:
            logger.warning(f"No markdown files found in topic: {topic_name}")
            self.stats['processed_topics'].append(topic_name)
            self.checkpoint.save(self.stats)
            return 0
        
        all_chunks = []
        
        for file_metadata in tqdm(md_files, desc=f"Processing {topic_name}", unit="file"):
            chunks = self.process_file(file_metadata)
            all_chunks.extend(chunks)
            self.stats['files_processed'] += 1
        
        if not all_chunks:
            logger.warning(f"No chunks created for topic: {topic_name}")
            self.stats['processed_topics'].append(topic_name)
            self.checkpoint.save(self.stats)
            return 0
        
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedder.generate_embeddings_batch(texts)
        self.stats['embeddings_generated'] += len(embeddings)
        
        logger.info(f"Inserting {len(all_chunks)} documents into Supabase...")
        inserted = self.vector_store.insert_documents(all_chunks, embeddings)
        self.stats['documents_inserted'] += inserted
        
        self.stats['topics_processed'] += 1
        self.stats['processed_topics'].append(topic_name)
        
        # Save checkpoint after each topic
        self.checkpoint.save(self.stats)
        
        return inserted
    
    def run(self, max_topics: Optional[int] = None):
        """Run the complete pipeline"""
        logger.info("\n" + "="*80)
        logger.info("AWS DOCUMENTATION VECTORIZATION PIPELINE (GitHub Actions)")
        logger.info("="*80)
        logger.info(f"Repository: {self.config.github_repository}")
        logger.info(f"Git SHA: {self.config.github_sha}")
        logger.info(f"Git Ref: {self.config.github_ref}")
        logger.info(f"Chunk size: {self.config.chunk_size}")
        logger.info(f"Chunk overlap: {self.config.chunk_overlap}")
        logger.info("="*80 + "\n")
        
        start_time = time.time()
        
        try:
            topics = self.fetcher.list_topic_folders()
            
            if max_topics:
                topics = topics[:max_topics]
                logger.info(f"Limited to first {max_topics} topics")
            
            for topic in topics:
                try:
                    self.process_topic(topic)
                except Exception as e:
                    logger.error(f"Error processing topic {topic['topicName']}: {e}")
                    self.stats['errors'] += 1
                    continue
            
            elapsed_time = time.time() - start_time
            self.print_statistics(elapsed_time)
            
            # Clean up checkpoint on success
            if Path(self.config.checkpoint_file).exists():
                Path(self.config.checkpoint_file).unlink()
                logger.info("Checkpoint file removed (pipeline completed successfully)")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.checkpoint.save(self.stats)
            raise
    
    def print_statistics(self, elapsed_time: float):
        """Print pipeline execution statistics"""
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("="*80)
        logger.info(f"Topics processed:      {self.stats['topics_processed']}")
        logger.info(f"Files processed:       {self.stats['files_processed']}")
        logger.info(f"Chunks created:        {self.stats['chunks_created']}")
        logger.info(f"Embeddings generated:  {self.stats['embeddings_generated']}")
        logger.info(f"Documents inserted:    {self.stats['documents_inserted']}")
        logger.info(f"Errors encountered:    {self.stats['errors']}")
        logger.info(f"Total execution time:  {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
        logger.info("="*80 + "\n")


def main():
    """Main entry point"""
    try:
        # Load configuration
        config = WorkflowConfig()
        
        # Get max_topics from environment (for manual workflow dispatch)
        max_topics_env = os.getenv('INPUT_MAX_TOPICS', '')
        max_topics = int(max_topics_env) if max_topics_env else None
        
        # Create and run pipeline
        pipeline = AWSDocumentationPipeline(config)
        pipeline.run(max_topics=max_topics)
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()