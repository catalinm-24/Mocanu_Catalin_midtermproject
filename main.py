import runpy
from pathlib import Path

def main():
    script = Path(__file__).parent / "midterm_from_notebook.py"
    runpy.run_path(str(script), run_name="__main__")

if __name__ == "__main__":
    main()
