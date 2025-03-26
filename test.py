from pathlib import Path

# List of module names
modules = [
    "initialize",
    "tcbf",
    "dpc",
    "direct_ptychography",
    "fft",
    "iterative_ptychography",
    "loss_and_direction",
    "multislice",
    "signal_extraction",
    "utils",
    "vaa",
]

# Create the reference folder if it doesn't exist
reference_dir = Path("docs/reference")
reference_dir.mkdir(parents=True, exist_ok=True)

# Generate each .md file with full import path
for module in modules:
    path = reference_dir / f"{module}.md"
    with open(path, "w") as f:
        f.write(f"# `pypty.{module}`\n\n::: pypty.{module}\n")
    print(f"✅ Created: {path}")
