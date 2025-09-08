#!/usr/bin/env python
"""
Initialize the policy database and create embeddings
Run this script once to set up the policy search system
"""

import sys
import os

# Add src to path
sys.path.append('src')

from src.tools import create_policy_embeddings

def main():
    print("🚀 Initializing Policy Database...")
    print("-" * 40)
    
    try:
        # Create embeddings for all policies
        result = create_policy_embeddings()
        
        if result.get('status') == 'success':
            print(f"✅ Success! Indexed {result.get('policies_indexed', 0)} policies")
            print("✅ Policy database is ready for RAG queries")
        else:
            print(f"⚠️ {result.get('message', 'Unknown status')}")
            
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure customer_policies.json exists in data/ folder")
        print("2. Check that ChromaDB is properly installed")
        print("3. Verify the embedding model path is correct")
        return 1
    
    print("-" * 40)
    print("🎉 Initialization complete!")
    return 0

if __name__ == "__main__":
    exit(main())