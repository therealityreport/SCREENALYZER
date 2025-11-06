#!/bin/bash
# Syntax validation for all Python task files

set -e

echo "🔍 Running syntax checks on all task files..."

TASKS_DIR="jobs/tasks"
ERRORS=0

for file in "$TASKS_DIR"/*.py; do
    if [ -f "$file" ]; then
        echo -n "Checking $(basename "$file")... "
        if python3 -m py_compile "$file" 2>/dev/null; then
            echo "✅"
        else
            echo "❌"
            python3 -m py_compile "$file"
            ERRORS=$((ERRORS + 1))
        fi
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo ""
    echo "✅ All task files passed syntax validation"
    exit 0
else
    echo ""
    echo "❌ $ERRORS file(s) failed syntax validation"
    exit 1
fi
