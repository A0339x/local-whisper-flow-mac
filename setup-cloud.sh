#!/bin/bash
# Convenience alias for the full-cloud edition. Same as: ./setup.sh --cloud
# (Most people should just run ./setup.sh — it asks which edition you want.)
exec "$(dirname "$0")/setup.sh" --cloud "$@"
