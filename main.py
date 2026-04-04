import os
import sys

def run_script(script_path):
    print(f"\n{'='*50}")
    print(f"Running {script_path}...")
    print(f"{'='*50}")
    result = os.system(f"{sys.executable} \"{script_path}\"")
    if result != 0:
        print(f"Error executing {script_path}. Exiting.")
        sys.exit(1)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, "src")
    
    scripts = [
        os.path.join(src_dir, "download_data.py"),
        os.path.join(src_dir, "preprocess.py"),
        os.path.join(src_dir, "train.py"),
        os.path.join(src_dir, "evaluate.py")
    ]
    
    for script in scripts:
        run_script(script)
        
    print("\nIDS Pipeline completed successfully!")

if __name__ == "__main__":
    main()
