"""
Streamlit Community Cloud Entry Point.
Delegates to app/app.py for standard root-level discovery.
"""

import os
import sys

# Ensure project root is at index 0 of sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.app import main

if __name__ == "__main__":
    main()
