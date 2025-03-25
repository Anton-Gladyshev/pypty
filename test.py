import os
import ast
import sys
from tqdm import tqdm

def get_module_name(filepath, library_dir):
    """
    Compute a module name from the file path relative to the library directory,
    including the library folder name as a prefix.
    For example, if the library is at /path/to/library and filepath is
    /path/to/library/moduleA.py, the module name becomes 'library.moduleA'.
    For subdirectories, path separators are replaced by dots.
    """
    library_name = os.path.basename(os.path.normpath(library_dir))
    rel_path = os.path.relpath(filepath, library_dir)
    module_base = os.path.splitext(rel_path)[0]  # remove .py extension
    module_name = module_base.replace(os.sep, ".")
    return f"{library_name}.{module_name}"

def parse_module(filepath):
    """
    Parse the Python file and extract:
      - defined_funcs: mapping of function names (via def) to their line numbers.
      - star_imports: list of module names star-imported (from moduleX import *).
      - function_calls: list of bare function calls (function name, line number).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filepath)
    defined_funcs = {}
    star_imports = []
    function_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined_funcs[node.name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                if node.module:
                    star_imports.append(node.module.lstrip("."))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function_calls.append((node.func.id, node.lineno))
    return defined_funcs, star_imports, function_calls

def main(library_dir):
    # Initialize progress bar for 4 steps
    pbar = tqdm(total=4, desc="Processing", ncols=80)

    # Step 1: Scan directory for Python modules.
    modules = {}
    for root, dirs, files in os.walk(library_dir):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                module_name = get_module_name(filepath, library_dir)
                defined_funcs, star_imports, function_calls = parse_module(filepath)
                modules[module_name] = {
                    "file": os.path.relpath(filepath, library_dir),
                    "defined_funcs": defined_funcs,
                    "star_imports": star_imports,
                    "function_calls": function_calls
                }
    pbar.update(1)

    # Step 2: Build star-import mapping.
    # For each module, list which other modules it star-imports (only within the library).
    star_imports_map = {}
    for mod_name, info in modules.items():
        star_imports_map[mod_name] = []
        for imported in info["star_imports"]:
            if imported in modules:
                star_imports_map[mod_name].append(imported)
            else:
                for candidate in modules:
                    if candidate.endswith(imported):
                        star_imports_map[mod_name].append(candidate)
                        break
    pbar.update(1)

    # Step 3: Analyze function calls for external usage.
    usage_by_module = {}  # mapping: module -> { source_module: [(function_name, lineno), ...] }
    for mod, info in modules.items():
        usage_by_module.setdefault(mod, {})
        local_defined = info["defined_funcs"]
        for func_name, lineno in info["function_calls"]:
            if func_name in local_defined:
                continue
            for imported_mod in star_imports_map.get(mod, []):
                if func_name in modules[imported_mod]["defined_funcs"]:
                    usage_by_module[mod].setdefault(imported_mod, []).append((func_name, lineno))
    pbar.update(1)

    # Step 4: Generate the final report and save it.
    output_lines = []
    for mod in sorted(usage_by_module.keys()):
        if usage_by_module[mod]:
            output_lines.append(f"Module: {modules[mod]['file']}")
            for src_mod in sorted(usage_by_module[mod].keys()):
                output_lines.append(f"\tUsages from {modules[src_mod]['file']}:")
                for func_name, lineno in sorted(usage_by_module[mod][src_mod], key=lambda x: x[1]):
                    output_lines.append(f"\t\tFunction {func_name} at line {lineno}")
            output_lines.append("")
    if not output_lines:
        output_lines.append("No external function usages via star-imports were detected.")

    output_file = os.path.join(library_dir, "module_usage_report.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python module_usage_report.py /path/to/your/library")
    library_dir = sys.argv[1]
    main(library_dir)
