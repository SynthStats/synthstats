#!/bin/sh
set -eu

if [ -d "/workspace/synthstats/src/synthstats" ]; then
  export PYTHONPATH="/workspace/synthstats/src${PYTHONPATH:+:$PYTHONPATH}"
elif [ -d "/workspace/synthstats/synthstats" ]; then
  export PYTHONPATH="/workspace/synthstats${PYTHONPATH:+:$PYTHONPATH}"
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SEARCH_DIR="$SCRIPT_DIR"
while [ "$SEARCH_DIR" != "/" ]; do
  if [ -d "$SEARCH_DIR/src/synthstats" ]; then
    export PYTHONPATH="$SEARCH_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
    break
  fi
  if [ -d "$SEARCH_DIR/synthstats" ]; then
    export PYTHONPATH="$SEARCH_DIR${PYTHONPATH:+:$PYTHONPATH}"
    break
  fi
  SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done

python -m synthstats.integrations.harbor.verifier
