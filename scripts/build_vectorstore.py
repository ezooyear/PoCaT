"""
Vector DB build script.
"""
import os
import shutil
import sys

from dotenv import load_dotenv

# Add project root to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from db.vectorstore import CHROMA_DIR, build_vectorstore


def main():
    _configure_stdout()

    print("=" * 60)
    print("Vector DB build script")
    print("=" * 60)

    if os.path.exists(CHROMA_DIR):
        print(f"\nRemoving existing Vector DB: {CHROMA_DIR}")
        try:
            shutil.rmtree(CHROMA_DIR)
        except PermissionError:
            _print_locked_db_help(CHROMA_DIR)
            sys.exit(1)

    print("\nBuilding Vector DB...\n")
    try:
        vectorstore = build_vectorstore()
    except PermissionError:
        _print_locked_db_help(CHROMA_DIR)
        sys.exit(1)

    if vectorstore:
        print("\n" + "=" * 60)
        print("Vector DB build completed.")
        print("Run the app with: streamlit run app.py --server.fileWatcherType none")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("Vector DB build failed.")
    print("Check whether PDF files exist under data/pdfs and whether the embedding model can be downloaded.")
    print("=" * 60)
    sys.exit(1)


def _configure_stdout():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _print_locked_db_help(chroma_dir: str):
    print("\n" + "=" * 60)
    print("Vector DB directory is locked by another process.")
    print(f"Locked path: {chroma_dir}")
    print("")
    print("Please close any running Streamlit or Python processes using this project, then try again.")
    print("Recommended order:")
    print("1. Stop Streamlit")
    print("2. Stop any Python process using Chroma DB")
    print("3. Re-run: .\\venv\\Scripts\\python.exe scripts\\build_vectorstore.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
