# Import Note

## Files are now in `scripts/biohash/` directory

### If you want to import BioHash modules from other locations:

**Option 1: Run from within the biohash directory**

```bash
cd /home/raduf/sandbox2/docuverse/scripts/biohash
python test_text_hashing.py
python biohash_text.py
```

**Option 2: Add biohash directory to Python path**

```python
import sys
sys.path.insert(0, '/home/raduf/sandbox2/docuverse/scripts/biohash')

from biohash_implementation import BioHash
from biohash_text import BioHashText, TfidfEmbedder
from biohash_docuverse import DocUVerseBioHash
```

**Option 3: Use relative imports (from scripts directory)**

```python
from biohash.biohash_implementation import BioHash
from biohash.biohash_text import BioHashText
from biohash.biohash_docuverse import DocUVerseBioHash
```

## Running Tests

All tests are designed to run from within the `biohash/` directory:

```bash
cd /home/raduf/sandbox2/docuverse/scripts/biohash

# Run any test
python validate_biohash.py
python test_biohash_simple.py
python test_text_hashing.py
python example_biohash_benchmark.py

# Run demos
python biohash_text.py
python biohash_docuverse.py
```

## Note for DocUVerse Integration

The `biohash_docuverse.py` module already handles the path correctly to access parent directory for DocUVerse imports.
