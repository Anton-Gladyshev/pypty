#!/bin/bash

# Install necessary dependencies
pip install pdoc3==0.11.6 mkdocs==1.6.1

# Generate Markdown docs from Python code
#pdoc3 --output-dir docs/reference --template-dir markdown_template pypty

# Serve docs locally and open browser automatically
mkdocs serve &
sleep 2
open http://127.0.0.1:8000
wait
