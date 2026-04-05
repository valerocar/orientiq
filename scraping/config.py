"""Configuration for the 3D Print Orientation Research Scraper."""

SUBREDDITS = [
    "3Dprinting",
    "FixMyPrint",
    "3dprintingbusiness",
    "prusa3d",
    "BambuLab",
    "resinprinting",
    "functionalprint",
    "slicing",
]

REDDIT_QUERIES = [
    "orientation",
    "auto-orient",
    "auto orient",
    "support minimization",
    "support removal",
    "support waste",
    "bed packing",
    "nesting",
    "batch printing",
    "print farm workflow",
    "model placement",
    "build plate layout",
    "overhang",
    "surface finish orientation",
    "bridge orientation",
    "print failure orientation",
]

REDDIT_POSTS_PER_QUERY = 100
REDDIT_COMMENTS_PER_POST = 20

GITHUB_REPOS = [
    "prusa3d/PrusaSlicer",
    "SoftFever/OrcaSlicer",
    "Ultimaker/Cura",
    "supermerill/SuperSlicer",
]

GITHUB_SEARCH_TERMS = [
    '"auto orient"',
    '"print orientation"',
    '"support generation"',
    '"nesting"',
    '"arrange"',
    '"bed packing"',
    '"model placement"',
    '"overhang angle"',
]

GITHUB_LABELS = [
    "feature-request",
    "enhancement",
    "auto-orient",
    "arrange",
]

YOUTUBE_QUERIES = [
    "print farm workflow",
    "print farm orientation",
    "3d print orientation tips",
    "auto orient 3d print",
    "print farm efficiency",
    "batch 3d printing workflow",
    "3d print nesting",
    "slicer auto arrange",
    "print farm day in the life",
]

YOUTUBE_PRIORITY_CHANNELS = [
    "Made with Layers",
    "3D Printing Nerd",
    "Uncle Jessy",
    "Thomas Sanladerer",
    "CNC Kitchen",
    "Slant 3D",
    "3D Musketeers",
    "Makers Muse",
]

YOUTUBE_MIN_DURATION = 180   # seconds (3 min)
YOUTUBE_MAX_DURATION = 3600  # seconds (60 min)
YOUTUBE_RESULTS_PER_QUERY = 20

# Paths
DATA_DIR = "data"
RAW_DIR = "data/raw"
LOG_FILE = "data/pipeline.log"
