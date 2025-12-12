import os
from pathlib import Path
import json

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def save_json(data, filepath):
    """Save dictionary as JSON"""
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70)

def format_number(num, decimals=2):
    """Format number with thousand separators"""
    return f"{num:,.{decimals}f}"