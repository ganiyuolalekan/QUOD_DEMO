# Configuration file for QUOD Task app

# App Settings
APP_TITLE = "QUOD Task - Documentation Processor"
APP_ICON = "📄"
MAX_FILE_SIZE_MB = 50
MAX_CONTENT_LENGTH = 100000  # characters

# OpenAI Settings
DEFAULT_MODEL = "gpt-4o-mini"  # Use gpt-4o as default for predictive output support
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4000

# Models that support predictive output (as of late 2024)
AVAILABLE_MODELS = [
    "gpt-4o",           # Latest GPT-4 Omni model with predictive output support
    "gpt-4o-mini",      # Smaller version with predictive output support  
    "gpt-4-turbo",      # Legacy model, may have limited predictive output
    "gpt-3.5-turbo"     # Fallback model, no predictive output
]

# Predictive output support by model
PREDICTIVE_OUTPUT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini"
]

# File Processing Settings
SUPPORTED_EXTENSIONS = {
    'jira': ['.md', '.txt', '.doc', '.docx', '.pdf'],
    'asciidoc': ['.adoc', '.adocx', '.md', '.txt', '.doc', '.docx']
}

# AsciiDoc Settings
ASCIIDOC_HEADING_LEVELS = 6
FASTDOC_MARKER = "FASTDOC"

# Processing Settings
PROCESSING_MODES = [
    "Update Existing Sections",
    "Add New Sections", 
    "Full Modification"
]

# UI Settings
SIDEBAR_WIDTH = 300
PREVIEW_MAX_LINES = 2000
DIFF_CONTEXT_LINES = 3
