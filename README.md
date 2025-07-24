# QUOD Task - Documentation Processor

A Streamlit application for processing Jira tickets and updating AsciiDoc documentation with AI assistance.

## Features

- **Multi-format File Support**: Process various file types (.md, .txt, .doc, .docx, .pdf, .adoc, .adocx)
- **Jira Ticket Analysis**: Extract TODO items, completed work, and key information from Jira tickets
- **AsciiDoc Processing**: Advanced AsciiDoc structure analysis and manipulation
- **AI-Powered Updates**: Use OpenAI to intelligently update documentation based on Jira context
- **FASTDOC Markers**: Automatic marking of new and modified sections
- **Side-by-Side Preview**: Compare original vs updated documentation
- **Export Options**: Download updated documents, diff reports, and processing summaries

## Installation

1. Clone or download this project to your local machine
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key (optional, fallback processing available):

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

   Or add it to Streamlit secrets in `.streamlit/secrets.toml`:

   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your browser to the displayed URL ([typically open here](http://localhost:8501))

3. Follow the application workflow:
   - **Upload Files**: Upload your Jira ticket file and AsciiDoc documentation
   - **Configure Processing**: Choose processing options and AI settings
   - **Process Documents**: Let the AI analyze and update your documentation
   - **Review Results**: Preview the updated documentation with highlighted changes
   - **Download**: Export the updated files and reports

## Supported File Types

### Jira Ticket Files

- `.md` - Markdown files
- `.txt` - Plain text files
- `.doc` / `.docx` - Word documents
- `.pdf` - PDF files

### Documentation Files

- `.adoc` / `.adocx` - AsciiDoc files
- `.md` - Markdown files
- `.txt` - Plain text files
- `.doc` / `.docx` - Word documents

## FASTDOC Markers

The application automatically adds FASTDOC markers to indicate changes:

- `// *FASTDOC* - New Section Added` - Marks new sections
- `// *FASTDOC* - Updated Section` - Marks beginning of modifications
- `// *FASTDOC* - End Update` - Marks end of modifications

## Processing Modes

1. **Update Existing Sections**: Modify existing content based on Jira information
2. **Add New Sections**: Add new sections for new features/requirements
3. **Full Document Rewrite**: Comprehensive update of the entire document

## AI Configuration

- **Model Selection**: Choose from GPT-4o, GPT-4o-mini, GPT-4-turbo, or GPT-3.5-turbo
- **Temperature**: Control creativity vs consistency (0.0 = consistent, 1.0 = creative)
- **Max Tokens**: Set maximum response length (1000-8000)

## Fallback Processing

If OpenAI API is not available, the application provides fallback processing using:

- Regex pattern matching for TODO/DONE items
- Simple text analysis for content extraction
- Basic section addition for updates

## File Structure

```bash
quod_task/
├── app.py                 # Main Streamlit application
├── file_processors.py     # File processing utilities
├── asciidoc_processor.py  # AsciiDoc-specific processing
├── openai_integrator.py   # OpenAI API integration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Example Workflow

1. **Jira Ticket Content**:

   ```bash
   JIRA-123: Implement user authentication
   
   TODO:
   - Add login endpoint
   - Implement JWT validation
   
   COMPLETED:
   - Database schema created
   - User model implemented
   ```

2. **Original AsciiDoc**:

   ```bash
   = API Documentation
   
   == Overview
   This document describes the API.
   
   == Authentication
   Currently no authentication is implemented.
   ```

3. **Updated AsciiDoc** (with FASTDOC markers):

   ```bash
   = API Documentation
   
   == Overview
   This document describes the API.
   
   // *FASTDOC* - Updated Section
   == Authentication
   The API now includes JWT-based authentication.
   
   === Login Endpoint
   The login endpoint accepts user credentials.
   // *FASTDOC* - End Update
   
   // *FASTDOC* - New Section Added
   === Pending Implementation
   * JWT token validation
   * Session management improvements
   ```

## Troubleshooting

### File Processing Issues

- Ensure files are not corrupted
- Check file size limits (PDFs can be large)
- For .doc files, convert to .docx format

### OpenAI API Issues

- Verify API key is correctly set
- Check API quota and billing
- Use fallback processing if API is unavailable

### AsciiDoc Formatting

- Ensure proper heading syntax (=, ==, ===, etc.)
- Check for consistent indentation
- Validate AsciiDoc syntax before processing

## Dependencies

- `streamlit` - Web application framework
- `openai` - OpenAI API client
- `PyPDF2` - PDF text extraction
- `python-docx` - Word document processing
- `chardet` - Character encoding detection
- `python-markdown` - Markdown processing support

## Contributing

This application is designed for the QUOD task. For modifications:

1. Test file processing with various formats
2. Validate AsciiDoc output syntax
3. Ensure FASTDOC markers are properly placed
4. Test both OpenAI and fallback processing modes

## License

This project is created for the QUOD task and follows the requirements specified in the task description.
