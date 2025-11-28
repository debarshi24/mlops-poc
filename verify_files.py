#!/usr/bin/env python3
"""Verify all Python files compile without syntax errors"""
import py_compile
import sys
from pathlib import Path

files_to_check = [
    "src/__init__.py",
    "src/model/__init__.py",
    "src/train.py",
    "src/preprocess.py",
    "src/evaluate.py",
    "src/utils.py",
    "src/model/inference.py",
    "ml_pipeline.py",
    "deploy_model.py"
]

errors = []
for file in files_to_check:
    try:
        py_compile.compile(file, doraise=True)
        print(f"✓ {file}")
    except py_compile.PyCompileError as e:
        print(f"✗ {file}: {e}")
        errors.append(file)

if errors:
    print(f"\n❌ {len(errors)} file(s) have syntax errors")
    sys.exit(1)
else:
    print(f"\n✅ All {len(files_to_check)} files are valid")
    sys.exit(0)
