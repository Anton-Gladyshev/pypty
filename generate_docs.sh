#!/bin/bash

# Install necessary dependencies
pip install --upgrade mkdocs-literate-nav myst-parser

# Serve docs locally and open browser automatically
mkdocs serve &
sleep 2
open http://127.0.0.1:8000
wait
