pip install pdoc3==0.11.6
pip install mkdocs==1.6.1
pdoc3 --output-dir docs/reference --template-dir markdown_template pypty
mkdocs serve
