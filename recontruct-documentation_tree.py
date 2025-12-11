import os
import shutil
from pathlib import Path

def restructure_aws_documentation(base_path):
    """
    Restructures AWS documentation by:
    1. Moving all .md files from doc_source to parent folder
    2. Removing non-.md files and doc_source folder
    """
    
    base_path = Path(base_path)
    documents_path = base_path / "documents"
    
    if not documents_path.exists():
        print(f"Error: {documents_path} does not exist!")
        return
    
    # Statistics
    stats = {
        'topics_processed': 0,
        'md_files_moved': 0,
        'files_deleted': 0,
        'folders_deleted': 0
    }
    
    # Iterate through each topic folder in documents
    for topic_folder in documents_path.iterdir():
        if not topic_folder.is_dir():
            continue
            
        print(f"\nProcessing topic: {topic_folder.name}")
        doc_source_path = topic_folder / "doc_source"
        
        if not doc_source_path.exists():
            print(f"  ⚠ No doc_source folder found, skipping...")
            continue
        
        # Move all .md files from doc_source to topic folder
        md_files = list(doc_source_path.glob("*.md"))
        
        if not md_files:
            print(f"  ⚠ No .md files found in doc_source")
        else:
            for md_file in md_files:
                destination = topic_folder / md_file.name
                
                # Handle file name conflicts
                if destination.exists():
                    print(f"  ⚠ File already exists: {md_file.name}, renaming to {md_file.stem}_from_doc_source.md")
                    destination = topic_folder / f"{md_file.stem}_from_doc_source.md"
                
                shutil.move(str(md_file), str(destination))
                print(f"  ✓ Moved: {md_file.name}")
                stats['md_files_moved'] += 1
        
        # Delete all remaining files in topic folder (not .md)
        for item in topic_folder.iterdir():
            if item.is_file() and item.suffix != '.md':
                print(f"  🗑 Deleting file: {item.name}")
                item.unlink()
                stats['files_deleted'] += 1
        
        # Delete doc_source folder and all its remaining contents
        if doc_source_path.exists():
            shutil.rmtree(doc_source_path)
            print(f"  🗑 Deleted doc_source folder")
            stats['folders_deleted'] += 1
        
        stats['topics_processed'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("RESTRUCTURING COMPLETE!")
    print("="*60)
    print(f"Topics processed: {stats['topics_processed']}")
    print(f"MD files moved: {stats['md_files_moved']}")
    print(f"Files deleted: {stats['files_deleted']}")
    print(f"Folders deleted: {stats['folders_deleted']}")
    print("="*60)

def create_backup(base_path):
    """Create a backup before restructuring"""
    base_path = Path(base_path)
    backup_path = base_path.parent / f"{base_path.name}_backup"
    
    if backup_path.exists():
        print(f"Backup already exists at {backup_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            return False
        shutil.rmtree(backup_path)
    
    print(f"Creating backup at {backup_path}...")
    shutil.copytree(base_path, backup_path)
    print("✓ Backup created successfully!")
    return True

if __name__ == "__main__":
    # Path to your aws-documentation-main folder
    AWS_DOCS_PATH = r"C:\Users\mouad\Documents\aws-documentation"
    
    
    print("AWS Documentation Restructuring Script")
    print("="*60)
    print(f"Target path: {AWS_DOCS_PATH}")
    print("\nThis script will:")
    print("1. Move all .md files from doc_source to parent folders")
    print("2. Delete all non-.md files")
    print("3. Remove doc_source folders")
    print("="*60)
    
    # Ask for confirmation
    response = input("\nCreate backup before proceeding? (y/n): ")
    if response.lower() == 'y':
        if not create_backup(AWS_DOCS_PATH):
            print("Backup failed or cancelled. Exiting...")
            exit(1)
    
    response = input("\nProceed with restructuring? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        exit(0)
    
    # Run the restructuring
    restructure_aws_documentation(AWS_DOCS_PATH)