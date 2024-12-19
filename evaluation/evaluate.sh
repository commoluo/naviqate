#!/bin/bash

# XVFB_PID=$(ps aux | grep '[X]vfb :99' | awk '{print $2}')

# if [ -z "$XVFB_PID" ]; then
#   echo "No Xvfb process found on display :99"
# else
#   kill $XVFB_PID
#   echo "Xvfb process on display :99 stopped"
# fi

# Xvfb :99 -screen 0 1920x1080x16 &
# XVFB_PID=$!

# export DISPLAY=:99

# python evaluate.py

# kill $XVFB_PID

#!/bin/bash

# Check if XQuartz is running
if pgrep -x "XQuartz" > /dev/null; then
  echo "XQuartz is already running."
else
  echo "Starting XQuartz..."
  open -a XQuartz
  sleep 2  # Allow time for XQuartz to initialize
fi

# Set DISPLAY environment variable
export DISPLAY=:0

# Run the Python evaluation script
python evaluate.py

# No need to manually kill XQuartz, as it runs as a background service on macOS
echo "Evaluation complete."
