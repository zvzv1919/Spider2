#!/bin/bash

# Script to stop all processes originating from paths containing a specified pattern
# This includes processes whose executable path or working directory contains the pattern

# Check for required argument
if [[ -z "$1" ]]; then
    echo "Usage: $0 <pattern>"
    echo "  Stops all processes with paths containing <pattern>"
    echo ""
    echo "Examples:"
    echo "  $0 zvzv1919"
    echo "  $0 ashley"
    exit 1
fi

PATTERN="$1"

echo "Finding processes with paths containing '$PATTERN'..."

# Get current script's PID and parent shell PID to avoid killing ourselves
CURRENT_PID=$$
PARENT_PID=$PPID

# Find all PIDs with executable or cwd containing the pattern
PIDS_TO_KILL=()

for pid in /proc/[0-9]*; do
    pid_num=$(basename "$pid")
    
    # Skip if we can't read the process info
    if [[ ! -r "$pid/exe" ]] && [[ ! -r "$pid/cwd" ]]; then
        continue
    fi
    
    # Skip our own process and parent shell
    if [[ "$pid_num" == "$CURRENT_PID" ]] || [[ "$pid_num" == "$PARENT_PID" ]]; then
        continue
    fi
    
    # Check executable path
    exe_path=$(readlink "$pid/exe" 2>/dev/null)
    cwd_path=$(readlink "$pid/cwd" 2>/dev/null)
    cmdline=$(tr '\0' ' ' < "$pid/cmdline" 2>/dev/null)
    
    if [[ "$exe_path" == *"$PATTERN"* ]] || [[ "$cwd_path" == *"$PATTERN"* ]] || [[ "$cmdline" == *"$PATTERN"* ]]; then
        PIDS_TO_KILL+=("$pid_num")
        echo "  Found PID $pid_num:"
        echo "    exe: $exe_path"
        echo "    cwd: $cwd_path"
        echo "    cmd: ${cmdline:0:100}..."
    fi
done

if [[ ${#PIDS_TO_KILL[@]} -eq 0 ]]; then
    echo "No matching processes found."
    exit 0
fi

echo ""
echo "Found ${#PIDS_TO_KILL[@]} process(es) to stop."
echo ""

# Send SIGTERM first (graceful shutdown)
echo "Sending SIGTERM to processes..."
for pid in "${PIDS_TO_KILL[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Terminating PID $pid"
        kill -TERM "$pid" 2>/dev/null
    fi
done

# Wait a moment for graceful shutdown
sleep 2

# Check if any processes are still running and send SIGKILL
echo ""
echo "Checking for remaining processes..."
for pid in "${PIDS_TO_KILL[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        echo "  PID $pid still running, sending SIGKILL"
        kill -9 "$pid" 2>/dev/null
    fi
done

echo ""
echo "Done."

