"""
Quick visualization regenerator for presentation prep.
Run this script to quickly regenerate all charts and graphs.
"""

import subprocess
import sys
import os

def main():
    print("QUICK VISUALIZATION REGENERATOR")
    print("=" * 40)
    
    # Check if required files exist
    required_files = ['detailed_results.json', 'generate_visualizations.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nRun these commands first:")
        print("   python complete_model_evaluation.py")
        return
    
    print("All required files found!")
    print("\nRegenerating visualizations...")
    
    # Run the visualization generator
    try:
        result = subprocess.run([sys.executable, 'generate_visualizations.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Visualizations generated successfully!")
            print("\nFiles ready for presentation:")
            
            # List generated files
            viz_files = [
                'confusion_matrix_heatmap.png',
                'model_performance_visualizations.png', 
                'visual_analysis_report.txt',
                'mathematical_analysis.txt'
            ]
            
            for file in viz_files:
                if os.path.exists(file):
                    size = os.path.getsize(file)
                    if size > 1024:
                        size_str = f"{size//1024} KB"
                    else:
                        size_str = f"{size} bytes"
                    print(f"   {file} ({size_str})")
                else:
                    print(f"   {file} (not found)")
                    
            print("\nReady for your presentation!")
            print("Check VISUALIZATION_GUIDE.md for presentation tips!")
            
        else:
            print("Error generating visualizations:")
            print(result.stderr)
            
    except Exception as e:
        print(f" Error running visualization generator: {e}")

if __name__ == "__main__":
    main()
