#!/bin/bash

set -e

# Check if DISPLAY is set
if [ -z "$DISPLAY" ]; then
    echo "ERROR: DISPLAY variable is not set!"
    echo ""
    echo "You need to reconnect with X11 forwarding enabled:"
    echo "  ssh -X pran4372@recuvmonsoon.int.colorado.edu  (slower, more secure)"
    echo "  OR (recommended for better performance):"
    echo "  ssh -YC pran4372@recuvmonsoon.int.colorado.edu  (faster, with compression)"
    echo ""
    exit 1
fi

# Test X11 connection
echo "Testing X11 connection..."
if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    echo "WARNING: Cannot connect to X server at $DISPLAY"
    echo "Attempting to fix DISPLAY variable..."
    # Try common X11 display formats
    if [[ "$DISPLAY" == localhost:* ]]; then
        # If it's localhost:10.0, try :10.0
        DISPLAY_NUM="${DISPLAY#localhost:}"
        export DISPLAY=":$DISPLAY_NUM"
        echo "Trying DISPLAY=$DISPLAY"
    fi
    # Test again
    if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        echo "ERROR: X11 connection failed. Please check:"
        echo "  1. You connected with 'ssh -X' or 'ssh -Y'"
        echo "  2. X11 server is running on your local machine"
        echo "  3. X11Forwarding is enabled in SSH config"
        exit 1
    fi
fi
echo "X11 connection OK: $DISPLAY"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv directory not found!"
    exit 1
fi

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Restore original environment variables (unset conflicting Qt paths)
# This ensures we use the venv's Qt libraries instead of system/conda ones
ORIG_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH

# Let venv use its own Qt library
# Find PyQt5 installation and get the Qt library path
PYQT5_QT_LIB=$(python -c "from PyQt5 import QtCore; import os; qt_lib = os.path.join(os.path.dirname(QtCore.__file__), 'Qt5', 'lib'); print(qt_lib if os.path.exists(qt_lib) else '')" 2>/dev/null)
if [ -n "$PYQT5_QT_LIB" ] && [ -d "$PYQT5_QT_LIB" ]; then
    # Prepend venv's Qt library to LD_LIBRARY_PATH
    if [ -n "$ORIG_LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="$PYQT5_QT_LIB:$ORIG_LD_LIBRARY_PATH"
    else
        export LD_LIBRARY_PATH="$PYQT5_QT_LIB"
    fi
    echo "Using Qt libraries from: $PYQT5_QT_LIB"
else
    # Restore original if we couldn't find venv Qt libraries
    if [ -n "$ORIG_LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="$ORIG_LD_LIBRARY_PATH"
    fi
fi

# Fix Qt plugin issues with OpenCV
PYQT5_PLUGINS=$(python -c "from PyQt5 import QtCore; import os; print(os.path.join(os.path.dirname(QtCore.__file__), 'Qt5', 'plugins'))" 2>/dev/null)
if [ -n "$PYQT5_PLUGINS" ] && [ -d "$PYQT5_PLUGINS/platforms" ]; then
    export QT_QPA_PLATFORM_PLUGIN_PATH="$PYQT5_PLUGINS/platforms"
    export QT_PLUGIN_PATH="$PYQT5_PLUGINS"
    echo "Using Qt plugins from: $PYQT5_PLUGINS/platforms"
fi

# Set Qt platform and disable OpenGL
export QT_QPA_PLATFORM=xcb
export QT_XCB_GL_INTEGRATION=none

export QT_OPENGL=software
export QT_QUICK_BACKEND=software
export LIBGL_ALWAYS_SOFTWARE=1

export QT_X11_NO_MITSHM=1

export QT_NO_GLIB=1

export QT_LOGGING_RULES="*.debug=false;qt.qpa.*=false"

export QT_XCB_FORCE_SOFTWARE_OPENGL=1

export QT_XCB_FORCE_SOFTWARE_OPENGL=1

# Check if x-anylabeling is installed
if ! command -v x-anylabeling &> /dev/null; then
    echo "ERROR: x-anylabeling not found in venv!"
    echo "Install it with: pip install x-anylabeling"
    exit 1
fi

# Ensure GroundingDINO tokenizer file exists (fixes model loading error)
# Quick check: skip if tokenizer file already exists and is valid
TOKENIZER_FILE=$(python -c "import anylabeling; from pathlib import Path; print(Path(anylabeling.__file__).parent / 'services' / 'auto_labeling' / 'configs' / 'bert_base_uncased_tokenizer.json')" 2>/dev/null)
if [ -n "$TOKENIZER_FILE" ] && [ -f "$TOKENIZER_FILE" ]; then
    # Quick validation: check if file is readable and not empty
    if [ -s "$TOKENIZER_FILE" ]; then
        echo "✓ GroundingDINO tokenizer already exists, skipping setup check"
    else
        echo "Checking GroundingDINO setup..."
        python "$SCRIPT_DIR/setup_groundingdino.py" || {
            echo "WARNING: Failed to setup GroundingDINO tokenizer. Model may not load correctly."
        }
    fi
else
    echo "Checking GroundingDINO setup..."
    python "$SCRIPT_DIR/setup_groundingdino.py" || {
        echo "WARNING: Failed to setup GroundingDINO tokenizer. Model may not load correctly."
    }
fi

# Default paths
TILES_DIR="${1:-outputs/tiles}"
LABELS_FILE="${2:-schemas/classes.txt}" # can be changed later for more detailed labels in the future!!

# Check if tiles directory exists, if not try to find the most recent one
if [ ! -d "$TILES_DIR" ]; then
    echo "WARNING: Tiles directory '$TILES_DIR' not found!"
    echo "Searching for tiles directories..."
    
    # Find all tiles directories, sorted by modification time (newest first)
    LATEST_TILES=$(find outputs -type d -name "tiles" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -n "$LATEST_TILES" ] && [ -d "$LATEST_TILES" ]; then
        echo "Found tiles directory: $LATEST_TILES"
        TILES_DIR="$LATEST_TILES"
    else
        echo "ERROR: No tiles directory found!"
        echo "Please create tiles first or specify a valid tiles directory:"
        echo "  ./run_anylabeling.sh <tiles_directory> [labels_file]"
        exit 1
    fi
fi

# Verify tiles directory is actually a directory and not empty
if [ ! -d "$TILES_DIR" ]; then
    echo "ERROR: '$TILES_DIR' is not a valid directory!"
    exit 1
fi

if [ ! -f "$LABELS_FILE" ]; then
    echo "WARNING: Labels file '$LABELS_FILE' not found!"
    echo "Trying alternative: schemas/classes.txt"
    if [ -f "schemas/classes.txt" ]; then
        LABELS_FILE="schemas/classes.txt"
    else
        echo "ERROR: No labels file found!"
        exit 1
    fi
fi


echo "Starting X-AnyLabeling..."
echo "  Tiles directory: $TILES_DIR"
echo "  Labels file: $LABELS_FILE"
echo "  DISPLAY: $DISPLAY"
echo ""

x-anylabeling "$TILES_DIR" --labels "$LABELS_FILE"

