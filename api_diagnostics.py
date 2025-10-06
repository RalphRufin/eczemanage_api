"""
API Diagnostic Script
=====================
Comprehensive diagnostics for EASI Severity Prediction API
Checks dependencies, models, file paths, and numpy compatibility issues
"""

import sys
import os
from pathlib import Path
import importlib.util

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def check_python_environment():
    """Check Python version and environment"""
    print_section("Python Environment")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"Current Working Directory: {os.getcwd()}")


def check_package_versions():
    """Check installed package versions"""
    print_section("Package Versions")
    
    packages = [
        'numpy',
        'tensorflow',
        'fastapi',
        'uvicorn',
        'pillow',
        'pandas',
        'sklearn',
        'pydantic',
    ]
    
    for package in packages:
        try:
            if package == 'pillow':
                import PIL
                print(f"✓ PIL (Pillow): {PIL.__version__}")
            elif package == 'sklearn':
                import sklearn
                print(f"✓ scikit-learn: {sklearn.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                print(f"✓ {package}: {version}")
        except ImportError as e:
            print(f"✗ {package}: NOT INSTALLED - {e}")
        except Exception as e:
            print(f"✗ {package}: ERROR - {e}")


def check_numpy_detailed():
    """Detailed numpy diagnostics"""
    print_section("NumPy Detailed Diagnostics")
    
    try:
        import numpy as np
        print(f"✓ NumPy Version: {np.__version__}")
        print(f"✓ NumPy Location: {np.__file__}")
        
        # Check for numpy._core
        try:
            import numpy._core
            print(f"✓ numpy._core exists: {numpy._core.__file__}")
        except ImportError:
            print("✗ numpy._core NOT FOUND (NumPy < 2.0)")
            print("  This is the main issue! NumPy 2.0+ required for numpy._core")
        
        # Check for numpy.core (old path)
        try:
            import numpy.core
            print(f"✓ numpy.core exists: {numpy.core.__file__}")
        except ImportError:
            print("✗ numpy.core NOT FOUND")
        
        # Check numpy configuration
        print(f"\nNumPy Configuration:")
        try:
            np.show_config()
        except:
            print("  Could not show numpy config")
            
    except ImportError as e:
        print(f"✗ NumPy NOT INSTALLED: {e}")
    except Exception as e:
        print(f"✗ NumPy ERROR: {e}")


def check_tensorflow():
    """Check TensorFlow installation and GPU support"""
    print_section("TensorFlow Diagnostics")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow Version: {tf.__version__}")
        print(f"✓ TensorFlow Location: {tf.__file__}")
        print(f"✓ Built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"✓ GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        
        # List devices
        devices = tf.config.list_physical_devices()
        print(f"\nAvailable Devices:")
        for device in devices:
            print(f"  - {device}")
            
    except ImportError as e:
        print(f"✗ TensorFlow NOT INSTALLED: {e}")
    except Exception as e:
        print(f"✗ TensorFlow ERROR: {e}")


def check_model_files():
    """Check for required model files"""
    print_section("Model Files Check")
    
    # Check Derm Foundation model paths
    print("\n1. Derm Foundation Model:")
    derm_paths = [
        "./derm_foundation/",
        "./",
        "./saved_model/",
        "./model/",
        "./derm-foundation/"
    ]
    
    found_derm = False
    for path in derm_paths:
        saved_model_pb = os.path.join(path, "saved_model.pb")
        if os.path.exists(saved_model_pb):
            print(f"  ✓ Found: {saved_model_pb}")
            print(f"    Size: {os.path.getsize(saved_model_pb)} bytes")
            found_derm = True
            
            # Check for variables folder
            variables_path = os.path.join(path, "variables")
            if os.path.exists(variables_path):
                print(f"    Variables folder: {variables_path}")
                var_files = os.listdir(variables_path)
                print(f"    Variable files: {len(var_files)}")
        else:
            print(f"  ✗ Not found: {saved_model_pb}")
    
    if not found_derm:
        print("\n  ⚠ WARNING: No Derm Foundation model found!")
    
    # Check EASI model
    print("\n2. EASI Model:")
    easi_path = './trained_model/easi_severity_model_derm_foundation_individual.pkl'
    if os.path.exists(easi_path):
        print(f"  ✓ Found: {easi_path}")
        print(f"    Size: {os.path.getsize(easi_path)} bytes")
        
        # Try to peek at pickle contents
        try:
            import pickle
            with open(easi_path, 'rb') as f:
                try:
                    model_data = pickle.load(f)
                    print(f"    Keys in model: {list(model_data.keys())}")
                    if 'keras_model_path' in model_data:
                        keras_path = model_data['keras_model_path']
                        print(f"    Keras model path: {keras_path}")
                        if os.path.exists(keras_path):
                            print(f"    ✓ Keras model exists: {keras_path}")
                        else:
                            print(f"    ✗ Keras model NOT FOUND: {keras_path}")
                except Exception as e:
                    print(f"    ✗ Error loading pickle: {e}")
        except ImportError:
            print("    ✗ pickle module not available")
    else:
        print(f"  ✗ Not found: {easi_path}")
        print(f"    Current directory: {os.getcwd()}")
        
        # Check if trained_model directory exists
        if os.path.exists('./trained_model/'):
            print(f"    trained_model/ exists. Contents:")
            for item in os.listdir('./trained_model/'):
                print(f"      - {item}")


def check_directory_structure():
    """Check directory structure"""
    print_section("Directory Structure")
    
    current_dir = os.getcwd()
    print(f"Current Directory: {current_dir}\n")
    
    # List all items in current directory
    items = os.listdir('.')
    print("Contents:")
    for item in sorted(items):
        path = os.path.join('.', item)
        if os.path.isdir(path):
            print(f"  📁 {item}/")
        else:
            size = os.path.getsize(path)
            print(f"  📄 {item} ({size} bytes)")


def test_pickle_load():
    """Test if pickle can load with current numpy"""
    print_section("Pickle Load Test")
    
    easi_path = './trained_model/easi_severity_model_derm_foundation_individual.pkl'
    
    if not os.path.exists(easi_path):
        print(f"✗ Model file not found: {easi_path}")
        return
    
    try:
        import pickle
        import numpy as np
        
        print(f"Attempting to load: {easi_path}")
        print(f"NumPy version: {np.__version__}")
        
        with open(easi_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("✓ Successfully loaded pickle file!")
        print(f"Model data keys: {list(model_data.keys())}")
        
    except ModuleNotFoundError as e:
        print(f"✗ Module not found: {e}")
        print("\n  DIAGNOSIS: The pickle file was saved with a newer NumPy version.")
        print("  SOLUTION: Upgrade numpy to version 2.0 or higher")
        print("  Command: pip install --upgrade numpy>=2.0")
        
    except Exception as e:
        print(f"✗ Error loading pickle: {e}")
        print(f"  Error type: {type(e).__name__}")


def check_sklearn():
    """Check scikit-learn and its compatibility"""
    print_section("Scikit-learn Diagnostics")
    
    try:
        import sklearn
        print(f"✓ scikit-learn Version: {sklearn.__version__}")
        
        # Check for common sklearn modules used in the model
        try:
            from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
            print("✓ MultiLabelBinarizer available")
            print("✓ StandardScaler available")
        except ImportError as e:
            print(f"✗ Import error: {e}")
            
    except ImportError:
        print("✗ scikit-learn NOT INSTALLED")


def provide_solutions():
    """Provide solutions based on diagnostics"""
    print_section("Recommended Solutions")
    
    print("""
Based on the error "No module named 'numpy._core'", here are the solutions:

1. UPGRADE NUMPY (Recommended):
   pip install --upgrade numpy>=2.0.0
   
   This is the cleanest solution as newer packages expect NumPy 2.0+

2. If NumPy 2.0 causes compatibility issues, RECREATE THE PICKLE:
   - Load the original model with the old NumPy version
   - Save it again with protocol 4 for better compatibility
   - Or rebuild the model from scratch

3. CHECK ALL DEPENDENCIES:
   pip install --upgrade tensorflow numpy pandas scikit-learn pillow fastapi uvicorn

4. CREATE FRESH VIRTUAL ENVIRONMENT:
   python -m venv fresh_env
   source fresh_env/bin/activate  # On Windows: fresh_env\\Scripts\\activate
   pip install -r requirements.txt

5. VERIFY PACKAGE COMPATIBILITY:
   pip list --outdated
   pip check

After upgrading, restart your API server.
""")


def main():
    """Run all diagnostics"""
    print("=" * 70)
    print(" EASI API DIAGNOSTIC TOOL")
    print(" Analyzing system configuration and dependencies...")
    print("=" * 70)
    
    check_python_environment()
    check_package_versions()
    check_numpy_detailed()
    check_tensorflow()
    check_sklearn()
    check_directory_structure()
    check_model_files()
    test_pickle_load()
    provide_solutions()
    
    print("\n" + "=" * 70)
    print(" Diagnostics Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()