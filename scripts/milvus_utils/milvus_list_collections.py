#!/usr/bin/env python3
"""
Quick script to list collections and their sizes in a Milvus file database.

Usage:
    python milvus_list_collections.py <database_file>
    python milvus_list_collections.py milvus_demo.db
"""

import sys
from pymilvus import connections, utility, Collection

def list_collections(db_file):
    """List all collections and their sizes in a Milvus file database."""
    
    # Connect to Milvus with file database
    print(f"Connecting to Milvus database: {db_file}")
    try:
        connections.connect(
            alias="default",
            uri=db_file
        )
        print("✓ Connected successfully\n")
    except Exception as e:
        print(f"✗ Error connecting to database: {e}")
        return
    
    try:
        # List all collections
        collections = utility.list_collections()
        
        if not collections:
            print("No collections found in database")
            return
        
        print(f"Found {len(collections)} collection(s):\n")
        print(f"{'Collection Name':<40} {'Number of Entities':>20}")
        print("=" * 62)
        
        total_entities = 0
        for collection_name in collections:
            try:
                # Load collection and get count
                collection = Collection(collection_name)
                collection.load()
                count = collection.num_entities
                total_entities += count
                
                print(f"{collection_name:<40} {count:>20,}")
                
            except Exception as e:
                print(f"{collection_name:<40} {'Error: ' + str(e):>20}")
        
        print("=" * 62)
        print(f"{'TOTAL':<40} {total_entities:>20,}")
        
    except Exception as e:
        print(f"✗ Error listing collections: {e}")
    finally:
        connections.disconnect("default")
        print("\n✓ Disconnected from database")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python milvus_list_collections.py <database_file>")
        print("Example: python milvus_list_collections.py milvus_demo.db")
        sys.exit(1)
    
    db_file = sys.argv[1]
    list_collections(db_file)

# Example Output:
# 
# $ python milvus_list_collections.py milvus_demo.db
# Connecting to Milvus database: milvus_demo.db
# ✓ Connected successfully
#
# Found 3 collection(s):
#
# Collection Name                            Number of Entities
# ==============================================================
# documents                                             150,234
# embeddings                                            150,234
# metadata                                               45,678
# ==============================================================
# TOTAL                                                346,146
#
# ✓ Disconnected from database
