import streamlit as st
import os
import tempfile
from pathlib import Path
import difflib
from typing import Dict, List, Optional, Tuple
import re

# Import our custom modules
from file_processors import FileProcessor, SupportedFileTypes
from asciidoc_processor import AsciiDocProcessor
from openai_integrator import OpenAIDocumentationUpdater

class QuodTaskApp:
    """Main Streamlit application for QUOD Task - Jira + AsciiDoc processing"""
    
    def __init__(self):
        self.file_processor = FileProcessor()
        self.asciidoc_processor = AsciiDocProcessor()
        self.openai_updater = OpenAIDocumentationUpdater()
        
        # Initialize session state
        if 'jira_content' not in st.session_state:
            st.session_state.jira_content = None
        if 'asciidoc_content' not in st.session_state:
            st.session_state.asciidoc_content = None
        if 'processed_content' not in st.session_state:
            st.session_state.processed_content = None
        if 'jira_filename' not in st.session_state:
            st.session_state.jira_filename = None
        if 'asciidoc_filename' not in st.session_state:
            st.session_state.asciidoc_filename = None
        if 'change_snapshots' not in st.session_state:
            st.session_state.change_snapshots = []
        if 'change_summary' not in st.session_state:
            st.session_state.change_summary = None
    
    def render_header(self):
        """Render application header with styling"""
        st.set_page_config(
            page_title="QUOD Task - Documentation Processor",
            page_icon=None,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .file-preview {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
            max-height: 300px;
            overflow-y: auto;
            color: #212529;
        }
        .processing-info {
            background-color: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 1rem;
            margin: 1rem 0;
            color: #0c5460;
        }
        .success-message {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 1rem;
            color: #155724;
        }
        .fastdoc-marker {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 3px;
            padding: 0.2rem 0.5rem;
            font-weight: bold;
            color: #856404;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="main-header">
            <h1>QUOD Task - Documentation Processor</h1>
            <p>Process Jira tickets and update AsciiDoc documentation with AI assistance</p>
        </div>
        """, unsafe_allow_html=True)
    
    def file_upload_section(self):
        """Handle file uploads and previews"""
        st.header("📁 File Upload Section")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Jira Ticket File")
            st.write("Upload your Jira ticket file (.md, .txt, .doc, .docx, .pdf)")
            
            jira_file = st.file_uploader(
                "Choose Jira file",
                type=['md', 'txt', 'doc', 'docx', 'pdf'],
                key="jira_uploader",
                help="This file contains Jira ticket information that will be used as context"
            )
            
            if jira_file:
                with st.spinner("Processing Jira file..."):
                    try:
                        content = self.file_processor.process_file(jira_file)
                        st.session_state.jira_content = content
                        st.session_state.jira_filename = jira_file.name
                        
                        st.success(f"Successfully loaded: {jira_file.name}")
                        
                        # Simple text preview only
                        with st.expander("👀 Preview Extracted Text", expanded=False):
                            preview_content = content[:800] + "..." if len(content) > 800 else content
                            st.text_area("Extracted Text", preview_content, height=200, disabled=True)
                    
                    except Exception as e:
                        st.error(f"Error processing Jira file: {str(e)}")
        
        with col2:
            st.subheader("AsciiDoc Documentation")
            st.write("Upload your AsciiDoc file (.adoc, .adocx, .md, .txt)")
            
            asciidoc_file = st.file_uploader(
                "Choose AsciiDoc file",
                type=['adoc', 'adocx', 'md', 'txt', 'doc', 'docx'],
                key="asciidoc_uploader",
                help="This file contains system documentation in AsciiDoc syntax"
            )
            
            if asciidoc_file:
                with st.spinner("Processing AsciiDoc file..."):
                    try:
                        content = self.file_processor.process_file(asciidoc_file)
                        st.session_state.asciidoc_content = content
                        st.session_state.asciidoc_filename = asciidoc_file.name
                        
                        st.success(f"Successfully loaded: {asciidoc_file.name}")
                        
                        # Simple text preview only
                        with st.expander("👀 Preview Extracted Text", expanded=False):
                            preview_content = content[:800] + "..." if len(content) > 800 else content
                            st.text_area("Extracted Text", preview_content, height=200, disabled=True)
                    
                    except Exception as e:
                        st.error(f"Error processing AsciiDoc file: {str(e)}")
        
        return st.session_state.jira_content is not None and st.session_state.asciidoc_content is not None
    

    

    
    def process_documents(self):
        """Process the documents using OpenAI with default configuration"""
        st.header("Document Processing")
        
        if not (st.session_state.jira_content and st.session_state.asciidoc_content):
            st.warning("Please upload both files before processing")
            return False
        
        # Use default configuration
        config = {
            'processing_mode': 'Full Modification',
            'include_todos': True,
            'preserve_formatting': True,
            'ai_model': 'gpt-4o-mini',
            'temperature': 0.0,
            'max_tokens': 15000
        }
        
        with st.expander("Processing Information", expanded=True):
            st.markdown(f"""
            <div class="processing-info">
                <h4>Processing Details:</h4>
                <ul>
                    <li><strong>Jira File:</strong> {st.session_state.jira_filename}</li>
                    <li><strong>AsciiDoc File:</strong> {st.session_state.asciidoc_filename}</li>
                    <li><strong>Processing Mode:</strong> {config['processing_mode']} (default)</li>
                    <li><strong>AI Model:</strong> {config['ai_model']} (default)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing with AI... This may take a few moments"):
                progress_bar = st.progress(0)
                
                try:
                    # Step 1: Generate structured JIRA context automatically
                    progress_bar.progress(10)
                    
                    try:
                        jira_info = self.openai_updater.extract_jira_structured_info(
                            st.session_state.jira_content
                        )
                        
                        if jira_info['success']:
                            st.session_state.jira_markdown_context = jira_info['markdown_content']
                        else:
                            st.session_state.jira_markdown_context = None
                    except Exception as e:
                        st.warning(f"⚠️ Error structuring JIRA context: {str(e)}, using original content")
                        st.session_state.jira_markdown_context = None
                    
                    # Step 2: Analyze Jira content
                    progress_bar.progress(20)
                    
                    jira_analysis = self.openai_updater.analyze_jira_content(
                        st.session_state.jira_content,
                        include_todos=config['include_todos']
                    )
                    
                    # Step 3: Analyze AsciiDoc structure
                    progress_bar.progress(40)
                    
                    asciidoc_structure = self.asciidoc_processor.analyze_structure(
                        st.session_state.asciidoc_content
                    )
                    
                    # Step 4: Generate diff and updates
                    progress_bar.progress(60)
                    
                    # Get JIRA markdown context if available
                    jira_markdown_context = getattr(st.session_state, 'jira_markdown_context', None)
                    
                    updated_content = self.openai_updater.update_documentation(
                        jira_content=jira_markdown_context if jira_markdown_context is not None else st.session_state.jira_content,
                        asciidoc_content=st.session_state.asciidoc_content,
                        config=config
                    )
                    
                    # Step 5: Extract change snapshots and generate summary
                    progress_bar.progress(80)
                    
                    # Extract change snapshots from the updated content
                    snapshots = self.openai_updater.extract_change_snapshots(
                        original_content=st.session_state.asciidoc_content,
                        updated_content=updated_content
                    )
                    
                    # Generate individual AI summaries for each change
                    for snapshot in snapshots:
                        ai_summary = self.openai_updater.generate_individual_change_summary(
                            original_section=snapshot['original'],
                            updated_section=snapshot['updated'],
                            change_type=snapshot['type'],
                            config=config
                        )
                        snapshot['ai_summary'] = ai_summary
                    
                    # Generate overall LLM summary of changes
                    jira_context = getattr(st.session_state, 'jira_markdown_context', None)
                    change_summary = self.openai_updater.generate_change_summary(
                        jira_content=jira_context if jira_context is not None else st.session_state.jira_content,
                        snapshots=snapshots,
                        config=config
                    )
                    
                    # Store results in session state
                    st.session_state.change_snapshots = snapshots
                    st.session_state.change_summary = change_summary
                    
                    progress_bar.progress(100)
                    st.session_state.processed_content = updated_content
                    return True
                
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.exception(e)
                    return False
        
        return st.session_state.processed_content is not None
    
    def display_results(self):
        """Display processing results and download options"""
        if not st.session_state.processed_content:
            return
        
        st.header("Processing Results")
        
        # Tabs for different views (removed Rendered Preview and AI Prompts)
        tab1, tab2, tab3 = st.tabs(["Side-by-Side Comparison", "Raw Output", "Change Snapshots"])
        
        with tab1:
            st.subheader("Before vs After Comparison")
            
            # Add display options
            col_opts = st.columns([1, 1, 1])
            with col_opts[0]:
                char_limit = st.number_input("Character limit for preview", 
                                           min_value=1000, 
                                           max_value=50000, 
                                           value=10000, 
                                           step=1000,
                                           help="Adjust character limit for side-by-side comparison")
            with col_opts[1]:
                show_full_content = st.checkbox("Show full content", 
                                               value=True,
                                               help="Show complete content (may be slow for very large files)")
            with col_opts[2]:
                if st.button("Refresh View"):
                    st.rerun()
            
            # Determine actual limit to use
            actual_limit = None if show_full_content else char_limit
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original")
                if actual_limit and len(st.session_state.asciidoc_content) > actual_limit:
                    original_preview = st.session_state.asciidoc_content[:actual_limit] + f"\n\n... (showing first {actual_limit:,} characters of {len(st.session_state.asciidoc_content):,} total)"
                else:
                    original_preview = st.session_state.asciidoc_content
                st.text_area("Original Content", original_preview, height=400, key="original_preview")
            
            with col2:
                st.markdown("### Updated")
                if actual_limit and len(st.session_state.processed_content) > actual_limit:
                    updated_preview = st.session_state.processed_content[:actual_limit] + f"\n\n... (showing first {actual_limit:,} characters of {len(st.session_state.processed_content):,} total)"
                else:
                    updated_preview = st.session_state.processed_content
                st.text_area("Updated Content", updated_preview, height=400, key="updated_preview")
            
            # Show diff statistics
            self.display_diff_stats()
        
        with tab2:
            st.subheader("Raw AsciiDoc Output")
            st.code(st.session_state.processed_content, language="asciidoc")
        
        with tab3:
            self.display_change_snapshots()
        
        # Download section
        self.download_section()
    

    
    def display_diff_stats(self):
        """Display detailed diff statistics"""
        original_lines = st.session_state.asciidoc_content.split('\n')
        updated_lines = st.session_state.processed_content.split('\n')
        
        # Generate diff
        diff = list(difflib.unified_diff(
            original_lines, 
            updated_lines, 
            fromfile='Original', 
            tofile='Updated', 
            lineterm=''
        ))
        
        added_lines = len([line for line in diff if line.startswith('+')])
        removed_lines = len([line for line in diff if line.startswith('-')])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Added Lines", added_lines, delta=added_lines)
        with col2:
            st.metric("Removed Lines", removed_lines, delta=-removed_lines)
        with col3:
            st.metric("FASTDOC Markers", st.session_state.processed_content.count("FASTDOC"))
        with col4:
            st.metric("Total Changes", added_lines + removed_lines)
    
    def display_change_snapshots(self):
        """Display change snapshots in a clean table format with 3 columns: Original, Updated, AI Summary"""
        st.subheader("Change Analysis & Snapshots")
        
        # Check if we have snapshots
        if not hasattr(st.session_state, 'change_snapshots'):
            st.info("No change snapshots available. Process documents to generate change analysis.")
            return
        
        snapshots = st.session_state.change_snapshots
        
        if not snapshots:
            st.info("No changes detected in the processed document.")
            return
        
        # # Overall summary section
        # if hasattr(st.session_state, 'change_summary') and st.session_state.change_summary:
        #     st.markdown("### Overall Change Summary")
        #     with st.expander("AI Analysis of All Changes", expanded=False):
        #         st.markdown(st.session_state.change_summary)
        
        # Statistics
        # st.markdown("### Change Statistics")
        modification_count = len([s for s in snapshots if s['type'] == 'modification'])
        new_section_count = len([s for s in snapshots if s['type'] == 'new_section'])
        removal_count = len([s for s in snapshots if s['type'] == 'removal'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Changes", len(snapshots))
        with col2:
            st.metric("Modifications", modification_count)
        with col3:
            st.metric("New Sections", new_section_count)
        with col4:
            st.metric("Removals", removal_count)
        
        # Main snapshots table
        st.markdown(f"### Change Snapshots Table ({len(snapshots)} changes)")
        
        # Display each snapshot in a table-like format
        for i, snapshot in enumerate(snapshots, 1):
            with st.container():
                # st.markdown(f"#### Change {i}: {self._get_change_type_emoji(snapshot['type'])} {snapshot['type'].replace('_', ' ').title()}")
                
                # Create 3 columns for the table layout
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.markdown("**Original**")
                    original_content = snapshot.get('original', 'N/A')
                    if original_content and original_content != "No corresponding section found (New content)":
                        # Show preview
                        preview = self._create_content_preview(original_content, 150)
                        st.markdown(f"*{preview}*")
                        
                        # Expandable full content
                        with st.expander("View Full Original Content"):
                            st.code(original_content, language="asciidoc")
                    else:
                        st.info("New content - no original version")
                
                with col2:
                    st.markdown("**Updated**")
                    updated_content = snapshot.get('updated', 'N/A')
                    if updated_content and updated_content != "Content removed":
                        # Show preview
                        preview = self._create_content_preview(updated_content, 150)
                        st.markdown(f"*{preview}*")
                        
                        # Expandable full content
                        with st.expander("View Full Updated Content"):
                            st.code(updated_content, language="asciidoc")
                    else:
                        st.warning("Content was removed")
                
                with col3:
                    st.markdown("**AI Summary**")
                    ai_summary = snapshot.get('ai_summary', 'Summary not available')
                    st.markdown(ai_summary)
                
                # Add separator between changes
                if i < len(snapshots):
                    st.markdown("---")
    
    def _get_change_type_emoji(self, change_type: str) -> str:
        """Get emoji for change type"""
        emoji_map = {
            'modification': '[MOD]',
            'new_section': '[NEW]',
            'removal': '[DEL]',
            'unchanged': '[UNCH]'
        }
        return emoji_map.get(change_type, '[CHG]')
    
    def _create_content_preview(self, content: str, max_length: int = 150) -> str:
        """Create a preview of content for display"""
        if not content or content.strip() in ['N/A', 'Content removed']:
            return content
        
        if len(content) <= max_length:
            return content
        
        # Try to find a good break point
        preview = content[:max_length]
        last_space = preview.rfind(' ')
        if last_space > max_length * 0.7:  # If we find a space in the last 30%
            preview = preview[:last_space]
        
        return preview + "..."
    
    def download_section(self):
        """Provide download options for processed content"""
        st.header("Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download updated AsciiDoc
            st.download_button(
                label="Download Updated AsciiDoc",
                data=st.session_state.processed_content,
                file_name=f"updated_{st.session_state.asciidoc_filename}",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Download diff report
            diff_report = self.generate_diff_report()
            st.download_button(
                label="Download Diff Report",
                data=diff_report,
                file_name="fastdoc_diff_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            # Download processing summary
            summary = self.generate_processing_summary()
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="fastdoc_processing_summary.md",
                mime="text/markdown",
                use_container_width=True
            )
    
    def generate_diff_report(self) -> str:
        """Generate a detailed diff report"""
        original_lines = st.session_state.asciidoc_content.split('\n')
        updated_lines = st.session_state.processed_content.split('\n')
        
        diff = list(difflib.unified_diff(
            original_lines, 
            updated_lines, 
            fromfile=f'Original: {st.session_state.asciidoc_filename}', 
            tofile=f'Updated: {st.session_state.asciidoc_filename}', 
            lineterm=''
        ))
        
        report = f"""FASTDOC - Diff Report
Generated: {st.session_state.get('processing_timestamp', 'Unknown')}

Files:
- Original: {st.session_state.asciidoc_filename}
- Jira Context: {st.session_state.jira_filename}

Changes Summary:
- FASTDOC Markers Added: {st.session_state.processed_content.count("FASTDOC")}
- Lines Modified: {len([line for line in diff if line.startswith('+')])
            + len([line for line in diff if line.startswith('-')])}

Detailed Diff:
{''.join(diff)}
"""
        return report
    
    def generate_processing_summary(self) -> str:
        """Generate processing summary"""
        return f"""# FASTDOC Processing Summary

## Input Files
- **Jira Ticket**: {st.session_state.jira_filename}
- **AsciiDoc Documentation**: {st.session_state.asciidoc_filename}

## Processing Results
- **FASTDOC Markers Added**: {st.session_state.processed_content.count("FASTDOC")}
- **Original Content Length**: {len(st.session_state.asciidoc_content)} characters
- **Updated Content Length**: {len(st.session_state.processed_content)} characters
- **Change Ratio**: {((len(st.session_state.processed_content) - len(st.session_state.asciidoc_content)) / len(st.session_state.asciidoc_content) * 100):.1f}%

## Key Changes
All modifications and new sections have been marked with **FASTDOC** markers for easy identification.

## Next Steps
1. Review the updated documentation
2. Verify all FASTDOC markers are appropriate
3. Integrate the changes into your documentation system
4. Update version control with the new content

---
Generated by QUOD Task - Documentation Processor
"""
    
    def sidebar_info(self):
        """Display sidebar information and controls"""
        with st.sidebar:
            st.header("Process Status")
            
            # File upload status
            jira_status = "✅ Loaded" if st.session_state.jira_content else "⏳ Pending"
            asciidoc_status = "✅ Loaded" if st.session_state.asciidoc_content else "⏳ Pending"
            processing_status = "✅ Complete" if st.session_state.processed_content else "⏳ Pending"
            
            st.markdown(f"""
            - **Jira File**: {jira_status}
            - **AsciiDoc File**: {asciidoc_status}
            - **Processing**: {processing_status}
            """)
            
            st.divider()
            
            # Quick stats
            if st.session_state.processed_content:
                st.header("📊 Quick Stats")
                fastdoc_count = st.session_state.processed_content.count("FASTDOC")
                st.metric("FASTDOC Markers", fastdoc_count)
                st.metric("Content Size", f"{len(st.session_state.processed_content):,} chars")
            
            st.divider()
            
            # Reset button
            if st.button("🔄 Reset All", use_container_width=True):
                for key in ['jira_content', 'asciidoc_content', 'processed_content', 
                           'jira_filename', 'asciidoc_filename', 'jira_markdown_context',
                           'change_snapshots', 'change_summary']:
                    if key in ['change_snapshots']:
                        st.session_state[key] = []
                    else:
                        st.session_state[key] = None
                st.rerun()
            
            st.divider()
            
            # Help section
            with st.expander("❓ Help & Info"):
                st.markdown("""
                ### Supported File Types
                - **Text**: .md, .txt
                - **Word**: .doc, .docx  
                - **AsciiDoc**: .adoc, .adocx
                - **PDF**: .pdf
                
                ### FASTDOC Markers
                All changes are marked with `FASTDOC` tags to indicate:
                - New sections added
                - Existing sections modified
                - Content updates from Jira tickets
                
                ### Processing Flow
                1. Upload both files
                2. Configure processing options
                3. Process with AI
                4. Review and download results
                """)
    
    def run(self):
        """Main application runner"""
        self.render_header()
        self.sidebar_info()
        
        # Main processing flow
        files_uploaded = self.file_upload_section()
        
        if files_uploaded:
            processing_completed = self.process_documents()
            
            if processing_completed:
                self.display_results()
        else:
            st.info("👆 Please upload both files to begin processing")


def main():
    """Main entry point"""
    app = QuodTaskApp()
    app.run()


if __name__ == "__main__":
    main()
