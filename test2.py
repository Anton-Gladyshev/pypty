import os
import re
import sys

def parse_report(report_path):
    """
    Parses the generated text report and returns a dictionary mapping.
    The structure is:
      {
         "moduleA.py": {
             "moduleB.py": [(funcName, line_number), ...],
             "moduleC.py": [(funcName, line_number), ...],
         },
         ...
      }
    """
    file_usage = {}
    current_module = None
    current_imported = None
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("Module:"):
                # e.g.: "Module: moduleA.py"
                current_module = line.split("Module:")[1].strip()
                file_usage[current_module] = {}
            elif line.strip().startswith("usages from"):
                # e.g.: "usages from moduleB.py:"
                m = re.search(r"usages from\s+(.+):", line.strip())
                if m:
                    current_imported = m.group(1).strip()
                    if current_module is not None:
                        file_usage[current_module][current_imported] = []
            elif line.strip().startswith("function"):
                # e.g.: "function functionA at line 42"
                m = re.search(r"function\s+(\w+)\s+at line\s+(\d+)", line.strip())
                if m and current_module is not None and current_imported is not None:
                    func = m.group(1)
                    lineno = int(m.group(2))
                    file_usage[current_module][current_imported].append((func, lineno))
    return file_usage

def modify_star_import_line(line, library_name):
    """
    If the line is a star-import of the form:
      from library.moduleX import *
    replace it with:
      from library import moduleX as librarymoduleX
    """
    pattern = rf"^from\s+{re.escape(library_name)}\.([A-Za-z0-9_.]+)\s+import\s+\*$"
    m = re.match(pattern, line.strip())
    if m:
        module_part = m.group(1)
        # Create alias by concatenating library_name and module_part with dots removed.
        alias = library_name + module_part.replace(".", "")
        return f"from {library_name} import {module_part} as {alias}\n"
    return line

def modify_function_call_in_line(line, func_name, alias):
    """
    Replace bare occurrences of func_name with alias.func_name.
    Only replaces the word when not already preceded by a dot.
    """
    # Use a regex replacement with a function.
    pattern = rf"\b{re.escape(func_name)}\b"
    def repl(match):
        start = match.start()
        # If the character before is a dot, do not change.
        if start > 0 and line[start-1] == '.':
            return match.group(0)
        return f"{alias}.{match.group(0)}"
    new_line = re.sub(pattern, repl, line)
    return new_line

def main(library_dir):
    # Determine the library name from the directory name.
    library_name = os.path.basename(os.path.normpath(library_dir))
    report_path = os.path.join(library_dir, "module_usage_report.txt")
    file_usage = parse_report(report_path)

    # For each file mentioned in the report, modify it.
    for module_file, imported_mapping in file_usage.items():
        file_path = os.path.join(library_dir, module_file)
        if not os.path.exists(file_path):
            print(f"Warning: File {module_file} not found in library directory.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # First, replace star-import lines.
        new_lines = []
        for line in lines:
            new_lines.append(modify_star_import_line(line, library_name))
        
        # Next, for each reported usage, modify the corresponding line.
        for imported_file, usages in imported_mapping.items():
            # Compute alias: take the base name of the imported file (without extension)
            base = os.path.basename(imported_file)
            mod_name_without_ext = os.path.splitext(base)[0]
            alias = library_name + mod_name_without_ext  # e.g., librarymoduleB
            for func_name, lineno in usages:
                index = lineno - 1  # Convert to 0-index
                if index < len(new_lines):
                    new_lines[index] = modify_function_call_in_line(new_lines[index], func_name, alias)
                else:
                    print(f"Warning: Line {lineno} not found in {module_file}.")

        # Write the modified content back to the file.
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("".join(new_lines))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python modify_modules.py /path/to/your/library")
    library_dir = sys.argv[1]
    main(library_dir)
