"""
Data Loading Module for Credit Default Risk Prediction
========================================================

Purpose:
    - Download LendingClub dataset from stable source (Zenodo)
    - Validate data integrity and structure
    - Perform initial quality checks
    - Save metadata for reproducibility

Author: Credit Risk Analytics Team
Date: December 2024
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset metadata
DATASET_CONFIG = {
    'url': 'https://zenodo.org/records/11295916/files/LC_loans_granting_model_dataset.csv',
    'expected_rows_min': 1_000_000,
    'expected_rows_max': 1_500_000,
    'expected_columns_min': 10,
    'expected_columns_max': 20,
    'target_column': 'Default',
    'expected_md5': None,  # Add if known for validation
}

# Directory structure
DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# File paths
RAW_DATA_PATH = RAW_DIR / "lending_club_loans.csv"
METADATA_PATH = DATA_DIR / "raw_data_metadata.json"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create required directory structure if not exists."""
    for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured directory: {directory}")


def calculate_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Calculate MD5 hash of file for integrity verification.
    
    Args:
        filepath: Path to file
        chunk_size: Size of chunks to read (default 8KB)
    
    Returns:
        MD5 hash as hex string
    """
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def format_bytes(bytes_val: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def print_section_header(title: str, width: int = 80):
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def download_data(
    url: str, 
    filepath: Path, 
    force: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Download dataset from URL with progress tracking and validation.
    
    Args:
        url: Source URL
        filepath: Destination file path
        force: Force re-download if file exists
    
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    
    # Check if file already exists
    if filepath.exists() and not force:
        print(f"✓ Data already exists at {filepath}")
        print(f"  Size: {format_bytes(filepath.stat().st_size)}")
        print(f"  Use force=True to re-download")
        return True, None
    
    print(f"⏳ Downloading dataset from Zenodo...")
    print(f"   URL: {url}")
    print(f"   Destination: {filepath}")
    
    try:
        # Make request with stream=True for large files
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size == 0:
            return False, "Could not determine file size"
        
        print(f"   Total size: {format_bytes(total_size)}")
        print(f"   Estimated time: ~2-5 minutes (depending on connection)")
        
        # Download with progress bar
        with open(filepath, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc='Downloading'
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"\n✓ Download complete!")
        print(f"  File saved: {filepath}")
        print(f"  Size: {format_bytes(filepath.stat().st_size)}")
        
        # Calculate MD5 for integrity
        print(f"\n⏳ Calculating file hash for integrity check...")
        md5_hash = calculate_md5(filepath)
        print(f"✓ MD5 Hash: {md5_hash}")
        
        return True, None
        
    except requests.exceptions.Timeout:
        return False, "Download timeout - please check your internet connection"
    
    except requests.exceptions.RequestException as e:
        return False, f"Download failed: {str(e)}"
    
    except IOError as e:
        return False, f"File write error: {str(e)}"
    
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data_structure(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate dataset structure and basic quality.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Tuple of (is_valid: bool, errors: list)
    """
    errors = []
    
    # Check row count
    if not (DATASET_CONFIG['expected_rows_min'] <= len(df) <= DATASET_CONFIG['expected_rows_max']):
        errors.append(
            f"Row count {len(df):,} outside expected range "
            f"[{DATASET_CONFIG['expected_rows_min']:,}, {DATASET_CONFIG['expected_rows_max']:,}]"
        )
    
    # Check column count
    if not (DATASET_CONFIG['expected_columns_min'] <= len(df.columns) <= DATASET_CONFIG['expected_columns_max']):
        errors.append(
            f"Column count {len(df.columns)} outside expected range "
            f"[{DATASET_CONFIG['expected_columns_min']}, {DATASET_CONFIG['expected_columns_max']}]"
        )
    
    # Check target column exists
    if DATASET_CONFIG['target_column'] not in df.columns:
        errors.append(f"Target column '{DATASET_CONFIG['target_column']}' not found")
    
    # Check for completely empty dataframe
    if df.empty:
        errors.append("DataFrame is completely empty")
    
    # Check for all-null columns
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        errors.append(f"Columns with all null values: {all_null_cols}")
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > len(df) * 0.01:  # More than 1% duplicates
        errors.append(f"High duplicate count: {duplicate_count:,} rows ({duplicate_count/len(df)*100:.2f}%)")
    
    is_valid = len(errors) == 0
    
    return is_valid, errors


def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive data quality analysis.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary containing quality metrics
    """
    
    quality_metrics = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'duplicate_rows': df.duplicated().sum(),
        },
        'data_types': {
            'numeric': len(df.select_dtypes(include=[np.number]).columns),
            'object': len(df.select_dtypes(include=['object']).columns),
            'datetime': len(df.select_dtypes(include=['datetime64']).columns),
            'categorical': len(df.select_dtypes(include=['category']).columns),
        },
        'missing_values': {
            'total_missing': df.isnull().sum().sum(),
            'percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'columns_with_missing': df.columns[df.isnull().any()].tolist(),
        },
        'target_distribution': None,
    }
    
    # Analyze target variable if exists
    if DATASET_CONFIG['target_column'] in df.columns:
        target_col = DATASET_CONFIG['target_column']
        value_counts = df[target_col].value_counts()
        quality_metrics['target_distribution'] = {
            'values': value_counts.to_dict(),
            'default_rate': df[target_col].mean() if df[target_col].dtype in [int, float] else None,
        }
    
    return quality_metrics


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load CSV data with optimized memory usage.
    
    Args:
        filepath: Path to CSV file
        sample_size: Optional number of rows to sample (for testing)
    
    Returns:
        Loaded DataFrame
    """
    
    print(f"⏳ Loading data from {filepath}...")
    
    try:
        # Load with low_memory=False to prevent dtype warnings
        if sample_size:
            df = pd.read_csv(filepath, nrows=sample_size, low_memory=False)
            print(f"✓ Loaded sample of {len(df):,} rows")
        else:
            df = pd.read_csv(filepath, low_memory=False)
            print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        return df
        
    except FileNotFoundError:
        print(f"✗ Error: File not found at {filepath}")
        sys.exit(1)
    
    except pd.errors.EmptyDataError:
        print(f"✗ Error: File is empty")
        sys.exit(1)
    
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        sys.exit(1)


# =============================================================================
# METADATA MANAGEMENT
# =============================================================================

def save_metadata(df: pd.DataFrame, quality_metrics: Dict):
    """
    Save dataset metadata for reproducibility.
    
    Args:
        df: Source DataFrame
        quality_metrics: Quality analysis results
    """
    
    metadata = {
        'dataset': {
            'source': DATASET_CONFIG['url'],
            'filename': RAW_DATA_PATH.name,
            'download_timestamp': pd.Timestamp.now().isoformat(),
            'md5_hash': calculate_md5(RAW_DATA_PATH) if RAW_DATA_PATH.exists() else None,
        },
        'dimensions': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': round(df.memory_usage(deep=True).sum() / (1024**2), 2),
        },
        'columns': {
            'names': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
        },
        'quality_metrics': quality_metrics,
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\n✓ Metadata saved to: {METADATA_PATH}")


def print_data_summary(df: pd.DataFrame, quality_metrics: Dict):
    """
    Print comprehensive data summary.
    
    Args:
        df: DataFrame to summarize
        quality_metrics: Quality metrics dictionary
    """
    
    print_section_header("DATA SUMMARY")
    
    # Basic info
    print("📊 BASIC INFORMATION:")
    print(f"  Rows: {quality_metrics['basic_info']['rows']:,}")
    print(f"  Columns: {quality_metrics['basic_info']['columns']}")
    print(f"  Memory: {quality_metrics['basic_info']['memory_usage_mb']:.2f} MB")
    print(f"  Duplicates: {quality_metrics['basic_info']['duplicate_rows']:,}")
    
    # Data types
    print("\n📋 DATA TYPES:")
    print(f"  Numeric: {quality_metrics['data_types']['numeric']}")
    print(f"  Text/Object: {quality_metrics['data_types']['object']}")
    print(f"  Datetime: {quality_metrics['data_types']['datetime']}")
    
    # Missing values
    print("\n🔍 MISSING VALUES:")
    print(f"  Total: {quality_metrics['missing_values']['total_missing']:,}")
    print(f"  Percentage: {quality_metrics['missing_values']['percentage']:.2f}%")
    print(f"  Columns affected: {len(quality_metrics['missing_values']['columns_with_missing'])}")
    
    # Target distribution
    if quality_metrics['target_distribution']:
        print("\n🎯 TARGET DISTRIBUTION:")
        dist = quality_metrics['target_distribution']
        for value, count in dist['values'].items():
            pct = (count / quality_metrics['basic_info']['rows']) * 100
            label = "Fully Paid" if value == 0 else "Default"
            print(f"  {label} ({value}): {count:,} ({pct:.2f}%)")
        
        if dist['default_rate'] is not None:
            print(f"  Default Rate: {dist['default_rate']*100:.2f}%")
    
    # Sample columns
    print(f"\n📝 SAMPLE COLUMNS (first 10):")
    for i, col in enumerate(df.columns[:10], 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"  {i:2d}. {col:30s} ({dtype}) - {non_null:,} non-null")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print_section_header("PHASE 1: DATA LOADING & VALIDATION")
    
    # Step 1: Setup
    print("🔧 Setting up directory structure...")
    ensure_directories()
    
    # Step 2: Download
    success, error = download_data(
        url=DATASET_CONFIG['url'],
        filepath=RAW_DATA_PATH,
        force=False
    )
    
    if not success:
        print(f"\n✗ Download failed: {error}")
        print("\n📋 Manual download instructions:")
        print(f"   1. Visit: {DATASET_CONFIG['url']}")
        print(f"   2. Download file to: {RAW_DATA_PATH}")
        print(f"   3. Re-run this script")
        sys.exit(1)
    
    # Step 3: Load
    print_section_header("LOADING DATA")
    df = load_data(RAW_DATA_PATH)
    
    # Step 4: Validate
    print_section_header("VALIDATING DATA")
    is_valid, errors = validate_data_structure(df)
    
    if not is_valid:
        print("✗ Data validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print("✓ Data validation passed!")
    
    # Step 5: Analyze quality
    print("\n⏳ Analyzing data quality...")
    quality_metrics = analyze_data_quality(df)
    
    # Step 6: Summary
    print_data_summary(df, quality_metrics)
    
    # Step 7: Save metadata
    save_metadata(df, quality_metrics)
    
    # Final message
    print_section_header("✓ PHASE 1 COMPLETE")
    print("Next steps:")
    print("  1. Review metadata: cat data/raw_data_metadata.json")
    print("  2. Run EDA: python src/2_exploratory_analysis.py")
    print("  3. Check quality report for any concerns")


if __name__ == "__main__":
    main()