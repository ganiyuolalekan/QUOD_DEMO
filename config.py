# Configuration file for QUOD Task app

# App Settings
APP_TITLE = "QUOD Task - Documentation Processor"
APP_ICON = "📄"
MAX_FILE_SIZE_MB = 50
MAX_CONTENT_LENGTH = 100000  # characters

# OpenAI Settings
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4000
AVAILABLE_MODELS = [
    "gpt-4o", 
    "gpt-4o-mini", 
    "gpt-4-turbo", 
    "gpt-3.5-turbo"
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
    "Full Document Rewrite"
]

# UI Settings
SIDEBAR_WIDTH = 300
PREVIEW_MAX_LINES = 2000
DIFF_CONTEXT_LINES = 3
