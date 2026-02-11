"""
Show top 30 most repeated categories in AI-generated articles (filtered for content categories)
"""

import pandas as pd
from collections import Counter

# Load dataset
df = pd.read_csv('combined_training_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Total AI-generated samples: {df['is_ai_flagged'].sum()}")

# Filter for AI-generated only
df_ai = df[df['is_ai_flagged'] == 1]
print(f"Using {len(df_ai)} AI-generated articles")

# First, show sample categories to understand format
print(f"\n" + "="*70)
print("SAMPLE CATEGORIES (first 5 AI articles):")
print("="*70)
for idx, cats in enumerate(df_ai['categories'].head(5), 1):
    print(f"\n{idx}. {cats}")

print(f"\n" + "="*70)
print(f"TOP 30 CATEGORIES (ALL, UNFILTERED)")
print("="*70)

# Parse all categories first
all_ai_categories = []
for cats in df_ai['categories'].dropna():
    if isinstance(cats, str):
        # Split by semicolon or comma
        if ';' in cats:
            split_cats = [c.strip() for c in cats.split(';') if c.strip()]
        elif ',' in cats:
            split_cats = [c.strip() for c in cats.split(',') if c.strip()]
        else:
            split_cats = [cats.strip()] if cats.strip() else []
        all_ai_categories.extend(split_cats)

all_counts = Counter(all_ai_categories)
print(f"\nTotal category entries: {len(all_ai_categories)}")
print(f"Total unique categories: {len(all_counts)}")
print(f"\nTop 30 (unfiltered):")
for i, (cat, count) in enumerate(all_counts.most_common(30), 1):
    pct = 100 * count / len(df_ai)
    print(f"{i:2d}. {cat:60s} {count:6d} ({pct:.1f}%)")

# Now filter for content categories only
print(f"\n" + "="*70)
print(f"TOP 30 CATEGORIES (CONTENT ONLY - FILTERED)")
print("="*70)

# More targeted filter - only remove obvious administrative tags
def is_content_category(cat):
    """Return True if category is about content/topic, not maintenance."""
    exclude_patterns = [
        'Articles with short description',
        'Short description',
        'Use dmy dates',
        'Use mdy dates',
        'CS1',
        'Webarchive',
        'Commons category',
        'Pages using',
        'Pages with',
        'Coordinates on Wikidata',
        'Official website',
        'Infobox',
        'Harv and Sfn',
        'EngvarB',
        'AC with',
        'All stub',
        'Wikipedia articles',
        'Articles needing',
        'Articles containing',
        'Articles lacking',
    ]
    
    for pattern in exclude_patterns:
        if pattern.lower() in cat.lower():
            return False
    return True

filtered_categories = [c for c in all_ai_categories if is_content_category(c)]
filtered_counts = Counter(filtered_categories)

print(f"\nAfter filtering:")
print(f"Total category entries: {len(filtered_categories)}")
print(f"Total unique categories: {len(filtered_counts)}")
print(f"\nTop 30 content categories:")
for i, (cat, count) in enumerate(filtered_counts.most_common(30), 1):
    pct = 100 * count / len(df_ai)
    print(f"{i:2d}. {cat:60s} {count:6d} ({pct:.1f}%)")
