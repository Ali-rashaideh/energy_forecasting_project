# run_pipeline.py
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Define root directory
    root_dir = Path(__file__).parent
    
    # Create necessary directories
    (root_dir / "data/raw").mkdir(parents=True, exist_ok=True)
    (root_dir / "data/processed").mkdir(parents=True, exist_ok=True)
    (root_dir / "results").mkdir(parents=True, exist_ok=True)

    # Execute each stage in sequence
    stages = [
        ("data_preparation", "python src/data_preparation.py"),
        ("feature_engineering", "python src/feature_engineering.py"),
        ("modeling", "python src/modeling.py"),
        ("evaluation", "python src/evaluation.py")
    ]
    
    for name, command in stages:
        print(f"\n{'='*50}")
        print(f"RUNNING STAGE: {name.upper()}")
        print(f"{'='*50}")
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"✅ {name} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error in {name}: {e}")
            sys.exit(1)
    
    print("\n" + "="*50)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    # Print final conclusion
    conclusion_path = root_dir / "results/evaluation_conclusion.txt"
    if conclusion_path.exists():
        print("\nFINAL CONCLUSION:")
        print("-"*50)
        with open(conclusion_path, "r") as f:
            print(f.read())
    else:
        print("\n⚠️ Could not find evaluation conclusion")

if __name__ == "__main__":
    main()