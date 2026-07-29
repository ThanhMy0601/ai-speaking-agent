#!/usr/bin/env bash
#
# Kill every agent worker holding a connection to LiveKit.
#
# `pkill -f "agent.py"` is not enough: livekit-agents runs each job in a
# multiprocessing child whose command line is `python -c from multiprocessing
# ...`, with no trace of agent.py in it. Those children outlive the parent,
# keep their websocket to livekit-server open, and stay registered as
# workers — so livekit happily hands the next job to a process running the
# code you thought you had just replaced.
#
# The symptom is confusing enough to burn an hour: your new worker registers,
# a session starts, and the job is assigned to a worker ID you have never
# seen. Matching on the port instead of the command line catches all of them.
set -euo pipefail

PORT="${LIVEKIT_PORT:-7880}"

pids=$(lsof -nP -iTCP:"$PORT" 2>/dev/null | grep -v LISTEN | grep -i python | awk '{print $2}' | sort -u || true)

if [ -z "$pids" ]; then
  echo "No agent processes connected to :$PORT"
  exit 0
fi

for pid in $pids; do
  echo "killing $pid — $(ps -p "$pid" -o command= 2>/dev/null | cut -c1-80)"
  kill -9 "$pid" 2>/dev/null || true
done

sleep 1
echo "done — $(lsof -nP -iTCP:"$PORT" 2>/dev/null | grep -ci python || echo 0) python connection(s) left"
