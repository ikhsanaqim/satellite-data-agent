#!/bin/bash
echo "--- BASH SCRIPT STARTED ---"
echo "Running Streamlit on port 7860..."
exec streamlit run app.py --server.port=7860 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false
