#!/usr/bin/env python3
"""
Setup script to ensure GroundingDINO tokenizer file exists.
This fixes the "No such file or directory" error when loading GroundingDINO models.
Not important for now.
"""

import os
from pathlib import Path

def setup_groundingdino_tokenizer():
    """Create the missing BERT tokenizer config file for GroundingDINO."""
    try:
        from transformers import AutoTokenizer
        from tokenizers import Tokenizer
    except ImportError as e:
        print(f"ERROR: Missing required packages: {e}")
        print("Please install: pip install transformers tokenizers")
        return False
    
    try:
        import anylabeling
        anylabeling_path = Path(anylabeling.__file__).parent
    except ImportError:
        print("ERROR: anylabeling package not found")
        return False
    
    configs_dir = anylabeling_path / "services" / "auto_labeling" / "configs"
    tokenizer_file = configs_dir / "bert_base_uncased_tokenizer.json"
    
    if tokenizer_file.exists():
        print(f" Tokenizer file already exists: {tokenizer_file}")
        try:
            Tokenizer.from_file(str(tokenizer_file))
            print(" Tokenizer file is valid")
            return True
        except Exception as e:
            print(f"⚠ Tokenizer file exists but is invalid: {e}")
            print("  Recreating...")
    
    # Create configs directory if it doesn't exist
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and save tokenizer
    print(f"Downloading BERT tokenizer from Hugging Face...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.backend_tokenizer.save(str(tokenizer_file))
        print(f" Tokenizer file created: {tokenizer_file}")
        
        # Verify it can be loaded
        Tokenizer.from_file(str(tokenizer_file))
        print(" Tokenizer file is valid and can be loaded")
        return True
    except Exception as e:
        print(f"ERROR: Failed to create tokenizer file: {e}")
        return False

if __name__ == "__main__":
    success = setup_groundingdino_tokenizer()
    exit(0 if success else 1)

