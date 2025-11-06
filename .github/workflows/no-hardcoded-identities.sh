#!/bin/bash
#
# CI Guard: No-Hardcoded-Identities
#
# Fails if cast member names appear in config files (except allowlisted locations).
# Ensures identity-agnostic pipeline - no per-person tuning.

set -e

CAST_NAMES="EILEEN|RINNA|KIM|KYLE|BRANDI|YOLANDA|LVP"

echo "🔍 Checking for hardcoded identity names in configs..."

# Check YAML configs (exclude comments and rationale text)
if grep -r -E "$CAST_NAMES" configs/ --include="*.yaml" --include="*.yml" | \
   grep -v "^#" | \
   grep -v "rationale:" | \
   grep -v "spot-check" | \
   grep -v ".DEPRECATED"; then
    echo "❌ FAIL: Found hardcoded identity names in config files"
    echo ""
    echo "Identities must NOT be hardcoded in configs."
    echo "Use auto-caps (computed per episode) instead of per_identity overrides."
    echo ""
    echo "See: docs/AUTO_CAPS_DESIGN.md"
    exit 1
fi

echo "✅ PASS: No hardcoded identities in configs"

# Check Python code (exclude allowlisted locations)
echo ""
echo "🔍 Checking Python code (excluding allowlisted locations)..."

ALLOWLIST=(
    "jobs/tasks/"              # One-off task scripts OK
    "tests/"                   # Test files OK
    "docs/"                    # Documentation OK
    "diagnostics/"             # Reports OK
    "app/lib/"                 # UI display OK (shows names in tables)
)

# Build grep exclude pattern
EXCLUDE_PATTERN=""
for path in "${ALLOWLIST[@]}"; do
    EXCLUDE_PATTERN="$EXCLUDE_PATTERN --exclude-dir=$path"
done

if grep -r -E "per_identity.*$CAST_NAMES|if.*==.*['\"]($CAST_NAMES)" \
   screentime/ $EXCLUDE_PATTERN --include="*.py" | \
   grep -v "# allowlist:" | \
   grep -v "Example:"; then
    echo "❌ FAIL: Found hardcoded identity logic in core pipeline code"
    echo ""
    echo "Core pipeline code (screentime/) must be identity-agnostic."
    echo "Use auto-caps or dynamic computation instead of hardcoded checks."
    echo ""
    echo "Allowlisted locations (OK to have names):"
    for path in "${ALLOWLIST[@]}"; do
        echo "  - $path"
    done
    exit 1
fi

echo "✅ PASS: No hardcoded identity logic in core code"
echo ""
echo "✅ All checks passed - pipeline is identity-agnostic"
