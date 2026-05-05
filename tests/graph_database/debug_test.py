"""Simple test to debug Neo4j setup."""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(project_root, 'src'))

print("Starting test...")

try:
    print("Importing Neo4j repository...")
    from repositories.graph import Neo4jRepository
    print("✓ Import successful")
    
    print("Creating repository instance...")
    repo = Neo4jRepository()
    print("✓ Repository created")
    
    print("Testing connection...")
    driver = repo.conn_manager._driver
    if driver:
        print("✓ Driver is active")
    else:
        print("✗ Driver is None")
        
except Exception as e:
    import traceback
    print(f"✗ Error: {e}")
    traceback.print_exc()

print("Test completed")
