<%!
  import re
%>\
# \${module.name}

\${module.docstring}

% for func in module.functions():
## \${func.name}

\${func.docstring}

\`\`\`python
\${func.source}
\`\`\`

% endfor

% for cls in module.classes():
## class \${cls.name}

\${cls.docstring}

% for meth in cls.methods():
### \${meth.name}

\${meth.docstring}

\`\`\`python
\${meth.source}
\`\`\`

% endfor
% endfor
EOF
