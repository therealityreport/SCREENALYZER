#!/bin/bash
# Verify unified navigation structure

set -e

echo "============================================================"
echo "Screanalyzer Navigation Verification"
echo "============================================================"
echo ""

# 1. Check no legacy labeler files
echo "✓ Checking for legacy labeler files..."
if find app -name "*labeler*" -type f | grep -v deprecated | grep -q .; then
    echo "  ❌ FAIL: Found legacy labeler files"
    find app -name "*labeler*" -type f | grep -v deprecated
    exit 1
fi
echo "  ✅ No legacy labeler files found"

# 2. Check only one pages directory
echo ""
echo "✓ Checking for multiple pages directories..."
PAGES_DIRS=$(find app -type d -name "pages" -not -path "*/deprecated/*" | wc -l)
if [ "$PAGES_DIRS" -ne 1 ]; then
    echo "  ❌ FAIL: Found $PAGES_DIRS pages directories (expected 1)"
    find app -type d -name "pages" -not -path "*/deprecated/*"
    exit 1
fi
echo "  ✅ Found exactly 1 pages directory"

# 3. Check canonical pages exist
echo ""
echo "✓ Checking canonical pages..."
EXPECTED_PAGES=(
    "1_📤_Upload.py"
    "2_🎭_CAST.py"
    "3_🗂️_Workspace.py"
    "4_📊_Analytics.py"
    "5_⚙️_Settings.py"
)

for page in "${EXPECTED_PAGES[@]}"; do
    if [ ! -f "app/pages/$page" ]; then
        echo "  ❌ FAIL: Missing page: $page"
        exit 1
    fi
    echo "  ✅ Found: $page"
done

# 4. Check no cross-page imports
echo ""
echo "✓ Checking for cross-page imports..."
if grep -r "from app.pages" app/pages/ --include="*.py" 2>/dev/null | grep -v "__pycache__" | grep -q .; then
    echo "  ❌ FAIL: Found cross-page imports"
    grep -r "from app.pages" app/pages/ --include="*.py" | grep -v "__pycache__"
    exit 1
fi
echo "  ✅ No cross-page imports found"

# 5. Check entry point exists
echo ""
echo "✓ Checking canonical entry point..."
if [ ! -f "app/Home.py" ]; then
    echo "  ❌ FAIL: app/Home.py not found"
    exit 1
fi
echo "  ✅ Found: app/Home.py"

# 6. Verify legacy files in deprecated
echo ""
echo "✓ Checking deprecated files..."
LEGACY_FILES=(
    "labeler.py"
    "all_faces_redesign.py"
    "pairwise_review_redesign.py"
    "cluster_split.py"
    "review_pages.py"
)

for file in "${LEGACY_FILES[@]}"; do
    if [ -f "deprecated/$file" ]; then
        echo "  ✅ $file moved to deprecated/"
    else
        echo "  ⚠️  $file not found in deprecated/ (may have been deleted)"
    fi
done

echo ""
echo "============================================================"
echo "✅ Navigation Verification: PASSED"
echo "============================================================"
echo ""
echo "Start the unified app with:"
echo "  streamlit run app/Home.py"
echo ""
