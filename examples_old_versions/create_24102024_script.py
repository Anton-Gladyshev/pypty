from pathlib import Path
import argparse
from jinja2 import Template
import pypty

p = argparse.ArgumentParser()
p.add_argument('--template', required=True)
cmdargs = p.parse_args()

with open('params.py', 'r') as f:
    define_params = f.read()
script_tmpl = Path(pypty.__file__).parent.parent / 'examples' / cmdargs.template
with script_tmpl.open() as f:
    template_content = f.read()

template = Template(template_content)
rendered_script = template.render(define_params=define_params)
with open('run.py', 'w') as f:
    f.write(rendered_script)

