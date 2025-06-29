#!/usr/bin/env python3
import re

def fix_punctuation_spacing(text):
    """Fix the odd spacing around punctuation in the DailyDialog dataset"""
    # Very specific check for the EXACT pattern " ' " (with spaces) and replace directly first
    text = text.replace(" ' ", "'")
    
    # 1. First, standardize ALL apostrophes to the same character - handle both curly (') and straight (') apostrophes
    # This is CRITICAL - we need to convert all apostrophe types to one standard form
    text = text.replace("'", "'").replace("`", "'").replace("'", "'").replace("'", "'")
    
    # 2. Remove spaces around apostrophes - this handles all cases at once
    # This pattern matches any apostrophe with optional spaces around it
    text = re.sub(r'\s*\'\s*', "'", text)
    
    return text

# Test cases
test_cases = [
    "don ' t",
    "He couldn ' t",
    "It ' s fine",
    "It's fine",
    "don't",
    "Can't",
    "Yeah, it ' s been ages!"
]

for test in test_cases:
    fixed = fix_punctuation_spacing(test)
    print(f"Original: '{test}'")
    print(f"Fixed:    '{fixed}'")
    print()
