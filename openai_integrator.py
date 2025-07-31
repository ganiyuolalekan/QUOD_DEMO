import os
from typing import Dict, List
import difflib
import re
from openai import OpenAI
import streamlit as st

from dotenv import load_dotenv
from config import PREDICTIVE_OUTPUT_MODELS

load_dotenv()


class OpenAIDocumentationUpdater:
    """Handles OpenAI integration for updating documentation"""
    
    def __init__(self):
        self.client = None
        self.last_system_prompt = None
        self.last_user_prompt = None
        self.last_response = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            # Try to get API key from environment variable
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                # If not in environment, try to get from Streamlit secrets
                try:
                    api_key = st.secrets.get("OPENAI_API_KEY")
                except:
                    pass
            
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                st.warning("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY environment variable or add it to Streamlit secrets.")
        
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if OpenAI client is available"""
        return self.client is not None
    
    def supports_predictive_output(self, model: str) -> bool:
        """Check if the specified model supports predictive output"""
        return model in PREDICTIVE_OUTPUT_MODELS
    
    def analyze_jira_content(self, jira_content: str, include_todos: bool = True) -> Dict:
        """
        Analyze Jira ticket content to extract key information
        
        Args:
            jira_content (str): Jira ticket content
            include_todos (bool): Whether to include TODO items
            
        Returns:
            Dict: Analyzed information from Jira ticket
        """
        if not self.is_available():
            return self._fallback_jira_analysis(jira_content, include_todos)
        
        try:
            system_prompt = """You are an expert at analyzing Jira tickets and extracting key information for documentation updates.

Analyze the provided Jira ticket content and extract:
1. Main task/feature description
2. Technical requirements
3. Implementation details
4. TODO items (if requested)
5. Completed work
6. Key changes made

Return your analysis in a structured format that can be used to update technical documentation."""
            
            user_prompt = f"""Analyze this Jira ticket content:

{jira_content}

Include TODO items: {include_todos}

Please provide a structured analysis focusing on information that would be relevant for updating technical documentation."""
            
            # Store prompts for display
            self.last_system_prompt = system_prompt
            self.last_user_prompt = user_prompt
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for analysis
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            # Store response for display
            self.last_response = analysis_text
            
            return {
                'success': True,
                'analysis': analysis_text,
                'raw_content': jira_content,
                'method': 'openai'
            }
        
        except Exception as e:
            st.error(f"Error analyzing Jira content with OpenAI: {str(e)}")
            return self._fallback_jira_analysis(jira_content, include_todos)
    
    def _fallback_jira_analysis(self, jira_content: str, include_todos: bool) -> Dict:
        """Fallback analysis using regex patterns"""
        analysis_parts = []
        
        # Extract TODO items
        if include_todos:
            todo_patterns = [
                r'TODO[:\s]+(.+)',
                r'TO DO[:\s]+(.+)',
                r'- \[ \] (.+)',  # Checkbox format
                r'In Progress[:\s]+(.+)',
                r'PENDING[:\s]+(.+)'
            ]
            
            todos = []
            for pattern in todo_patterns:
                matches = re.findall(pattern, jira_content, re.IGNORECASE | re.MULTILINE)
                todos.extend(matches)
            
            if todos:
                analysis_parts.append(f"TODO Items:\n" + "\n".join(f"- {todo.strip()}" for todo in todos[:5]))
        
        # Extract completed items
        done_patterns = [
            r'DONE[:\s]+(.+)',
            r'COMPLETED[:\s]+(.+)',
            r'- \[x\] (.+)',  # Checked checkbox
            r'FIXED[:\s]+(.+)',
            r'RESOLVED[:\s]+(.+)'
        ]
        
        completed = []
        for pattern in done_patterns:
            matches = re.findall(pattern, jira_content, re.IGNORECASE | re.MULTILINE)
            completed.extend(matches)
        
        if completed:
            analysis_parts.append(f"Completed Items:\n" + "\n".join(f"- {item.strip()}" for item in completed[:5]))
        
        # Extract key sections
        sections = re.split(r'\n\s*\n', jira_content)
        if len(sections) > 1:
            analysis_parts.append(f"Key Content Sections: {len(sections)} sections identified")
        
        analysis_text = "\n\n".join(analysis_parts) if analysis_parts else "Basic content analysis completed."
        
        return {
            'success': True,
            'analysis': analysis_text,
            'raw_content': jira_content,
            'method': 'fallback'
        }
        
    def _get_mode_specific_instructions(self, processing_mode: str) -> str:
        """Get specific instructions based on processing mode"""
        
        if processing_mode == "Update Existing Sections":
            return """- ONLY update existing sections of the documentation with new information from the Jira ticket
-- DO NOT add any new sections or headings
-- Focus on modifying existing content to incorporate Jira information
-- If the Jira ticket mentions functionality that doesn't have an existing section, incorporate it into the most relevant existing section
-- There should be at least 3 sections updated in the documentation"""
        elif processing_mode == "Add New Sections":
            return """- ONLY add new sections to the documentation for functionality described in the Jira ticket
-- DO NOT modify existing sections unless absolutely necessary for consistency
-- Create new headings and sections for new features, components, or functionality mentioned in Jira
-- Maintain the document structure and add new sections in appropriate locations
-- There should be at least 2 new sections added to the documentation"""
        elif processing_mode == "Full Modification":
            return """- Both update existing sections AND add new sections as needed
-- Update existing documentation with new information from the Jira ticket
-- Add new sections to the documentation for new functionality that doesn't fit in existing sections but is mentioned in the jira ticket
-- Create comprehensive updates that can modify the entire document structure
-- Make meaningful changes that improve the documentation completeness
-- There should be at least 1 new sections added/removed/modified in the documentation due to the jira ticket content"""
        else:
            return """- Follow standard documentation update practices
-- Make changes that best reflect the Jira ticket information"""
            
    def update_documentation(self, jira_content: str, asciidoc_content: str, config: Dict) -> str:
        """
        Update AsciiDoc documentation based on Jira content using OpenAI predictive output
        
        Args:
            jira_content (str): Jira ticket content
            asciidoc_content (str): Original AsciiDoc content
            config (Dict): Configuration options
            
        Returns:
            str: Updated AsciiDoc content
        """
        if not self.is_available():
            return self._fallback_documentation_update(jira_content, asciidoc_content, config)
        
        try:
            # System prompt defines the AI's role and capabilities
            system_prompt = """You are an expert in AsciiDoc documentation modification.

CRITICAL REQUIREMENTS:
1. You MUST make meaningful changes to the documentation content that includes updating the documentation with new information confirmed from Jira ticket content you receive. 
2. The confirmed changes (which should be re-written text) should be marked as "MODIFIED", and the changes that were included should be marked as "ADDED".
3. All updated sections should be different from the section on the original document, using the jira input as context to update/modify that section
4. All sections that will receive FASTDOC markers must contain actual changes, not just copies

Your task:
1. Analyze the Jira ticket for specific information that needs to be integrated/added/modified in the documentation
2. Identify existing sections that need to be updated based on Jira content
3. Create new sections only if the Jira ticket introduces completely new functionality
4. For UPDATED sections: Modify the actual content by:
   - Adding new implementation details from Jira (ADDED)
   - Updating descriptions with new requirements (MODIFIED)
   - Adding new parameters, methods, or features mentioned in Jira (ADDED)
   - Incorporating TODO items as future enhancements (ADDED)
   - Including completed work as current functionality (ADDED)

CHANGE EXAMPLES:
- If Jira mentions "added JWT authentication", update auth sections in the asciidoc documentation to include JWT details
- If Jira lists new API endpoints, add them to the API of the asciidoc documentation
- If Jira describes bug fixes, update the affected sections of the asciidoc documentation with corrected information
- If Jira includes new configuration options, add them to configuration sections of the asciidoc documentation

FASTDOC MARKER REQUIREMENTS:
You MUST add FASTDOC markers to identify changes in your output:
- Wrap MODIFIED sections (updated existing content) with:
  // *FASTDOC* - Start Modification
  [modified content here]
  // *FASTDOC* - End Modification

- Wrap ADDED sections (completely new content) with:
  // *FASTDOC* - Start Modification  
  [new content here]
  // *FASTDOC* - End Modification

- For REMOVED content, add a comment where the content was removed:
  // *FASTDOC* - Content Removed (description of what was removed)

MARKER RULES:
- Only add markers around content that has actually changed
- Do NOT add markers around unchanged content
- Ensure markers are on their own lines
- Use AsciiDoc comment syntax (//) for markers
- Be precise - only mark the specific paragraphs/sections that changed

FORMAT REQUIREMENTS:
- Use proper AsciiDoc heading syntax (=, ==, ===, etc.)
- Maintain existing document structure, only parts of the document that needs change or a new section should be amended
- Preserve all original content that is still relevant
- Add FASTDOC markers directly in your output to identify all changes
- Ensure the full context from the jira ticket is included/reflected in the output

VALIDATION:
Before finishing, ensure every section you modified/added contains information that was NOT in the original document and is properly marked with FASTDOC markers."""
            
            # User prompt with modification instructions based on Jira content
            modification_prompt = f"""Based on the following JIRA ticket information, please modify the AsciiDoc documentation:

JIRA TICKET INFORMATION TO INTEGRATE:
{jira_content}

MODIFICATION INSTRUCTIONS:
Using the Jira Ticket Information, follow the instructions below to update the documentation:
{self._get_mode_specific_instructions(config.get('processing_mode', 'Update Existing Sections'))}

PROCESSING CONFIGURATION:
- Mode: {config.get('processing_mode', 'Update Existing Sections')}
- Include TODOs: {config.get('include_todos', True)}
- Preserve Formatting: {config.get('preserve_formatting', True)}

INSTRUCTIONS:
1. Maintain the asciidoc format from the original asciidoc documentation, and do not modify it's formatting to a standard format... instead, use the jira ticket content to update the documentation information.
2. Carefully read the Jira ticket to understand what changes need to be made
3. Identify which sections of the current documentation need updates based on the Jira content
4. Make actual changes to those sections by incorporating Jira information
5. Ensure that updated sections contain new information not present in the original document
6. Return the complete updated AsciiDoc document
7. Adjust as many sections of the asciidoc documentation that the jira ticket context addresses, you can add new sections if needed.

IMPORTANT: Do not simply copy existing sections. Every section that gets updated must contain actual changes derived from the Jira ticket content.

Respond only with the updated AsciiDoc content, with no markdown formatting."""
            
            # Store prompts for display
            self.last_system_prompt = system_prompt
            self.last_user_prompt = f"{modification_prompt}\n\nORIGINAL ASCIIDOC CONTENT:\n{asciidoc_content}"
            
            model = config.get('ai_model', 'gpt-4o-mini')
            
            # Use predictive output if the model supports it
            if self.supports_predictive_output(model):
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": modification_prompt},
                        {"role": "user", "content": asciidoc_content}
                    ],
                    prediction={
                        "type": "content",
                        "content": asciidoc_content
                    },
                    max_tokens=config.get('max_tokens', 4000),
                    temperature=config.get('temperature', 0.3)
                )
            else:
                st.warning(f"⚠️ Model {model} doesn't support predictive output. Using standard processing.")
                # Fallback to standard processing with combined prompt
                combined_prompt = f"{modification_prompt}\n\nCURRENT ASCIIDOC DOCUMENTATION:\n{asciidoc_content}"
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": combined_prompt}
                    ],
                    max_tokens=config.get('max_tokens', 4000),
                    temperature=config.get('temperature', 0.3)
                )
            
            updated_content = response.choices[0].message.content
            
            # Store response for display
            self.last_response = updated_content
            
            # Clean up any potential markdown formatting that might have been added
            updated_content = self._clean_asciidoc_output(updated_content)
            
            return updated_content
        
        except Exception as e:
            st.error(f"Error updating documentation with OpenAI: {str(e)}")
            return self._fallback_documentation_update(jira_content, asciidoc_content, config)
    
    def extract_change_snapshots(self, original_content: str, updated_content: str) -> List[Dict]:
        """
        Extract changed sections with FASTDOC markers and create clean snapshots for table display
        
        Args:
            original_content (str): Original content before changes
            updated_content (str): Content with FASTDOC markers
            
        Returns:
            List[Dict]: List of change snapshots with original, updated, and AI summary
        """
        snapshots = []
        updated_lines = updated_content.split('\n')
        
        i = 0
        while i < len(updated_lines):
            line = updated_lines[i].strip()
            
            # Look for start of modification marker
            if line.startswith('// *FASTDOC* - Start Modification'.replace('*', '_')):
                content_lines = []
                i += 1
                
                # Collect content between start and end markers
                while i < len(updated_lines):
                    if updated_lines[i].strip().startswith('// *FASTDOC* - End Modification'.replace('*', '_')):
                        break
                    content_lines.append(updated_lines[i])
                    i += 1
                else:
                    # No end marker found, skip this section
                    i += 1
                    continue
                
                updated_section = '\n'.join(content_lines).strip()
                
                # Find the corresponding section in original content
                original_section = self._find_related_original_section(original_content, updated_section)
                
                # Determine if this is a modification or new section
                if original_section:
                    change_type = 'modification' if original_section.strip() != updated_section.strip() else 'unchanged'
                else:
                    change_type = 'new_section'
                    original_section = "No corresponding section found (New content)"
                
                # Create snapshot for table display
                snapshot = {
                    'original': original_section,
                    'updated': updated_section,
                    'type': change_type,
                    'id': len(snapshots) + 1
                }
                snapshots.append(snapshot)
                
            # Look for content removed marker
            elif line.startswith('// *FASTDOC* - Content Removed'.replace('*', '_')):
                removed_desc = line.replace('// *FASTDOC* - Content Removed'.replace('*', '_')).strip()
                if removed_desc.startswith('(') and removed_desc.endswith(')'):
                    removed_desc = removed_desc[1:-1]
                
                snapshot = {
                    'original': removed_desc,
                    'updated': "Content removed",
                    'type': 'removal',
                    'id': len(snapshots) + 1
                }
                snapshots.append(snapshot)
            
            i += 1
        
        return snapshots
    
    def _find_related_original_section(self, original_content: str, updated_section: str) -> str:
        """
        Find the related section in original content that corresponds to the updated section
        
        Args:
            original_content (str): Original document content
            updated_section (str): Updated section content
            
        Returns:
            str: Related original section or None if not found (indicating new section)
        """
        if not updated_section.strip():
            return None
        
        updated_lines = [line.strip() for line in updated_section.split('\n') if line.strip()]
        if not updated_lines:
            return None
        
        # Strategy 1: Look for exact heading matches in AsciiDoc
        for line in updated_lines[:3]:  # Check first 3 lines for headings
            if line.startswith('=') and len(line) > 2:  # AsciiDoc heading
                heading_text = line.lstrip('=').strip()
                if heading_text in original_content:
                    # Found the heading, extract the original section
                    return self._extract_section_by_heading(original_content, heading_text)
        
        # Strategy 2: Look for similar content blocks using key phrases
        # Find the best matching paragraph/section in original content
        best_match = ""
        best_similarity = 0
        
        # Split original into logical sections (by double newlines or headings)
        original_sections = self._split_into_sections(original_content)
        
        for section in original_sections:
            if len(section.strip()) > 20:  # Only consider substantial sections
                similarity = self._calculate_text_similarity(updated_section, section)
                if similarity > best_similarity and similarity > 0.4:  # Higher threshold for better matching
                    best_similarity = similarity
                    best_match = section.strip()
        
        return best_match if best_match else None
    
    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into logical sections"""
        # Split by AsciiDoc headings and double newlines
        lines = content.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            if line.startswith('=') and len(line.strip()) > 2:  # New heading
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = [line]
                else:
                    current_section.append(line)
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        # Also split by double newlines for non-heading sections
        additional_sections = []
        for section in sections:
            if not any(line.startswith('=') for line in section.split('\n')):
                additional_sections.extend([s.strip() for s in section.split('\n\n') if s.strip()])
            else:
                additional_sections.append(section)
        
        return additional_sections
    
    def _extract_section_by_heading(self, content: str, heading_text: str) -> str:
        """Extract a section from content based on heading"""
        lines = content.split('\n')
        section_lines = []
        in_section = False
        current_heading_level = 0
        
        for line in lines:
            if heading_text in line and line.startswith('='):
                in_section = True
                current_heading_level = len(line) - len(line.lstrip('='))
                section_lines.append(line)
                continue
            
            if in_section:
                # Check if we hit another heading of same or higher level
                if line.startswith('='):
                    new_heading_level = len(line) - len(line.lstrip('='))
                    if new_heading_level <= current_heading_level:
                        break
                section_lines.append(line)
        
        return '\n'.join(section_lines).strip()
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic text similarity between two strings"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def generate_individual_change_summary(self, original_section: str, updated_section: str, change_type: str, config: Dict) -> str:
        """
        Generate an AI summary for an individual change
        
        Args:
            original_section (str): Original content section
            updated_section (str): Updated content section  
            change_type (str): Type of change (modification, new_section, removal)
            config (Dict): Configuration options
            
        Returns:
            str: AI-generated summary of the change
        """
        if not self.is_available():
            return f"Change type: {change_type}. OpenAI not available for detailed summary."
        
        try:
            if change_type == 'removal':
                system_prompt = "You are an expert at analyzing documentation changes. Summarize what content was removed and its significance."
                user_prompt = f"Content that was removed:\n{original_section}\n\nProvide a brief summary of what was removed and why this might be significant."
            
            elif change_type == 'new_section':
                system_prompt = "You are an expert at analyzing documentation changes. Summarize new content that was added."
                user_prompt = f"New content that was added:\n{updated_section}\n\nProvide a brief summary of what was added and its purpose."
            
            else:  # modification
                system_prompt = "You are an expert at analyzing documentation changes. Compare the original and updated content to summarize what changed."
                user_prompt = f"""Compare these two content sections and summarize what changed:

ORIGINAL:
{original_section}

UPDATED:
{updated_section}

Provide a brief summary of the key changes made and their significance."""

            response = self.client.chat.completions.create(
                model=config.get('ai_model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,  # Keep summaries concise
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def _create_content_preview(self, content: str, max_length: int = 100) -> str:
        """Create a preview of content for display"""
        if len(content) <= max_length:
            return content
        
        # Try to find a good break point
        preview = content[:max_length]
        last_space = preview.rfind(' ')
        if last_space > max_length * 0.7:  # If we find a space in the last 30%
            preview = preview[:last_space]
        
        return preview + "..."
    
    def generate_change_summary(self, jira_content: str, snapshots: List[Dict], config: Dict) -> str:
        """
        Generate an LLM summary of what changed and why it reflects the Jira ticket
        
        Args:
            jira_content (str): Original Jira ticket content
            snapshots (List[Dict]): List of change snapshots
            config (Dict): Configuration options
            
        Returns:
            str: Summary of changes
        """
        if not self.is_available() or not snapshots:
            return "No changes detected or OpenAI not available."
        
        try:
            # Prepare snapshot content for analysis
            snapshot_text = ""
            for i, snapshot in enumerate(snapshots, 1):
                snapshot_text += f"\n=== Change {i} ({snapshot['type']}) ===\n"
                snapshot_text += f"Location: Lines {snapshot['line_start']}-{snapshot['line_end']}\n"
                snapshot_text += f"Content: {snapshot['content']}\n"
            
            system_prompt = """You are an expert at analyzing documentation changes and their relationship to Jira tickets.

Your task is to:
1. Analyze the changes made to the documentation
2. Explain how each change relates to the Jira ticket content
3. Provide a clear, concise summary of what was modified/added/removed
4. Validate that the changes accurately reflect the Jira requirements

Format your response with:
- **Summary**: Overall description of changes
- **Change Analysis**: Detailed breakdown of each change and its purpose
- **Jira Alignment**: How the changes address the Jira ticket requirements
- **Impact**: What these changes mean for the documentation users"""

            user_prompt = f"""Please analyze the following documentation changes and explain how they relate to the Jira ticket:

JIRA TICKET CONTENT:
{jira_content}

DOCUMENTATION CHANGES MADE:
{snapshot_text}

Provide a comprehensive analysis of what changed and why these changes accurately reflect the Jira ticket requirements."""

            response = self.client.chat.completions.create(
                model=config.get('ai_model', 'gpt-4o-mini'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating change summary: {str(e)}"
    
    def _fallback_documentation_update(self, jira_content: str, asciidoc_content: str, config: Dict) -> str:
        """Fallback documentation update using simple text processing"""
        
        # Create a basic update by appending relevant Jira information
        updated_content = asciidoc_content
        
        # Extract TODO items from Jira
        if config.get('include_todos', True):
            todo_patterns = [
                r'TODO[:\s]+(.+)',
                r'TO DO[:\s]+(.+)', 
                r'- \[ \] (.+)',
                r'In Progress[:\s]+(.+)'
            ]
            
            todos = []
            for pattern in todo_patterns:
                matches = re.findall(pattern, jira_content, re.IGNORECASE | re.MULTILINE)
                todos.extend(matches)
            
            if todos:
                todo_section = "\n\n== Updates from Jira Ticket\n\n"
                todo_section += "=== Pending Tasks\n\n"
                for todo in todos[:5]:  # Limit to 5 items
                    todo_section += f"* {todo.strip()}\n"
                
                updated_content += todo_section
        
        # Extract completed items
        done_patterns = [
            r'DONE[:\s]+(.+)',
            r'COMPLETED[:\s]+(.+)',
            r'- \[x\] (.+)',
            r'FIXED[:\s]+(.+)'
        ]
        
        completed = []
        for pattern in done_patterns:
            matches = re.findall(pattern, jira_content, re.IGNORECASE | re.MULTILINE)
            completed.extend(matches)
        
        if completed:
            completed_section = "\n=== Completed Work\n\n"
            for item in completed[:5]:
                completed_section += f"* {item.strip()}\n"
            
            updated_content += completed_section
        
        return updated_content
    
    def _generate_diff(self, original: str, updated: str) -> str:
        """Generate diff between two text versions"""
        original_lines = original.split('\n')
        updated_lines = updated.split('\n')
        
        diff = difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile='original',
            tofile='updated',
            lineterm=''
        )
        
        return '\n'.join(diff)
    
    def _clean_asciidoc_output(self, content: str) -> str:
        """Clean up potential markdown formatting in AsciiDoc output"""
        
        # Remove asciidoc code block wrapper that sometimes gets added
        content = re.sub(r'^```asciidoc\s*\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n```\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```\s*\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n```\s*$', '', content, flags=re.MULTILINE)
        
        # Convert markdown headings to AsciiDoc if any slipped through
        content = re.sub(r'^#{6}\s+(.+)$', r'====== \1', content, flags=re.MULTILINE)
        content = re.sub(r'^#{5}\s+(.+)$', r'===== \1', content, flags=re.MULTILINE)
        content = re.sub(r'^#{4}\s+(.+)$', r'==== \1', content, flags=re.MULTILINE)
        content = re.sub(r'^#{3}\s+(.+)$', r'=== \1', content, flags=re.MULTILINE)
        content = re.sub(r'^#{2}\s+(.+)$', r'== \1', content, flags=re.MULTILINE)
        content = re.sub(r'^#{1}\s+(.+)$', r'= \1', content, flags=re.MULTILINE)
        
        # Convert markdown code blocks to AsciiDoc (but not the wrapper ones)
        content = re.sub(r'```(\w+)?\n(.*?)\n```', r'[source,\1]\n----\n\2\n----', content, flags=re.DOTALL)
        
        # Convert markdown bold/italic to AsciiDoc
        content = re.sub(r'\*\*(.+?)\*\*', r'*\1*', content)  # Bold
        content = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'_\1_', content)  # Italic
        
        return content
    
    def get_last_prompts(self) -> Dict:
        """Get the last used AI prompts and response for display/editing"""
        return {
            'system_prompt': self.last_system_prompt,
            'user_prompt': self.last_user_prompt,
            'response': self.last_response,
            'has_prompts': self.last_system_prompt is not None
        }
    
    def update_prompts(self, system_prompt: str, user_prompt: str) -> None:
        """Update the prompts for custom processing"""
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
    
    def reprocess_with_custom_prompts(self, system_prompt: str, user_prompt: str, config: Dict) -> str:
        """Reprocess documentation with custom prompts using predictive output"""
        if not self.is_available():
            st.error("❌ OpenAI client not available for custom prompt processing")
            return ""
        
        try:
            # Store the custom prompts
            self.last_system_prompt = system_prompt
            self.last_user_prompt = user_prompt
            
            model = config.get('ai_model', 'gpt-4o-mini')
            
            # Extract original AsciiDoc content from session state for predictive output
            original_content = ""
            if hasattr(st.session_state, 'asciidoc_content') and st.session_state.asciidoc_content:
                original_content = st.session_state.asciidoc_content
                
                # Check if model supports predictive output and content structure is compatible
                if (self.supports_predictive_output(model) and 
                    "ORIGINAL ASCIIDOC CONTENT:" in user_prompt):
                    
                    parts = user_prompt.split("ORIGINAL ASCIIDOC CONTENT:", 1)
                    modification_instructions = parts[0].strip()
                    if len(parts) > 1:
                        # Extract the original content from the prompt
                        original_content = parts[1].strip()
                    
                    # Use predictive output with separated prompts
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": modification_instructions},
                            {"role": "user", "content": original_content}
                        ],
                        prediction={
                            "type": "content",
                            "content": original_content
                        },
                        max_tokens=config.get('max_tokens', 4000),
                        temperature=config.get('temperature', 0.3)
                    )
                else:
                        
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=config.get('max_tokens', 4000),
                        temperature=config.get('temperature', 0.3)
                    )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=config.get('max_tokens', 4000),
                    temperature=config.get('temperature', 0.3)
                )
            
            updated_content = response.choices[0].message.content
            self.last_response = updated_content
            
            # Clean up output
            updated_content = self._clean_asciidoc_output(updated_content)
            
            return updated_content
            
        except Exception as e:
            st.error(f"❌ Error reprocessing with custom prompts: {str(e)}")
            return ""

    def get_usage_stats(self) -> Dict:
        """Get API usage statistics (placeholder)"""
        # This would require tracking API calls
        return {
            'api_calls_made': 0,
            'tokens_used': 0,
            'estimated_cost': 0.0,
            'available': self.is_available()
        }

    def extract_jira_structured_info(self, jira_content: str) -> Dict:
        """
        Extract structured JIRA ticket information and format as markdown
        
        Args:
            jira_content (str): Raw JIRA ticket content
            
        Returns:
            Dict: Structured JIRA information and markdown formatted content
        """
        if not self.is_available():
            return self._fallback_jira_structured_extraction(jira_content)
        
        try:
            system_prompt = """You are an expert at parsing JIRA ticket content and extracting structured information.

Extract the following information from the JIRA ticket content and format it as a structured markdown document:

**Required Fields to Extract:**
- Ticket Key/Name (e.g., PROJ-123)
- Title/Summary
- Parent Ticket Name (if any)
- Project
- Issue Type (Bug, Task, Story, etc.)
- Status (To Do, In Progress, Done, etc.)
- Priority (High, Medium, Low, etc.)
- Reporter
- Assignee
- Sprint (if mentioned)
- Sprint Team
- Story Points (if mentioned)
- Fixed Version (if mentioned)
- Components
- Repositories (if mentioned)
- Time Spent (if mentioned)

**Output Format:**
Create a markdown document with the following structure:

```markdown
# JIRA Ticket Analysis

## Ticket Information
- **Key:** [ticket key]
- **Title:** [title]
- **Parent Ticket:** [parent or N/A]
- **Project:** [project name]
- **Type:** [issue type]
- **Status:** [current status]
- **Priority:** [priority level]
- **Reporter:** [reporter name]
- **Assignee:** [assignee name]
- **Sprint:** [sprint name or N/A]
- **Sprint Team:** [team name or N/A]
- **Story Points:** [points or N/A]
- **Fixed Version:** [version or N/A]
- **Components:** [components or N/A]
- **Repositories:** [repositories or N/A]
- **Time Spent:** [time or N/A]

## Detailed Description
[Provide detailed description of all discovered information from the ticket]

## Sub-tasks
| Key | Summary | Assignees |
|-----|---------|-----------|
| [key] | [summary] | [assignees] |

## Comments
- [Comment 1]
- [Comment 2]
```

If information is not available in the content, use "N/A" or leave empty appropriately."""

            user_prompt = f"""Please extract structured information from this JIRA ticket content and format it as markdown:

{jira_content}

Provide the complete markdown document following the template structure."""

            # Store prompts for display
            self.last_system_prompt = system_prompt
            self.last_user_prompt = user_prompt

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )

            markdown_content = response.choices[0].message.content

            # Store response for display
            self.last_response = markdown_content

            return {
                'success': True,
                'markdown_content': markdown_content,
                'raw_content': jira_content,
                'method': 'openai'
            }

        except Exception as e:
            st.error(f"Error extracting JIRA structured info with OpenAI: {str(e)}")
            return self._fallback_jira_structured_extraction(jira_content)

    def _fallback_jira_structured_extraction(self, jira_content: str) -> Dict:
        """Fallback JIRA structured extraction using regex patterns"""
        import re
        
        # Extract basic information using regex patterns
        info = {
            'ticket_key': 'N/A',
            'title': 'N/A',
            'parent_ticket': 'N/A',
            'project': 'N/A',
            'issue_type': 'N/A',
            'status': 'N/A',
            'priority': 'N/A',
            'reporter': 'N/A',
            'assignee': 'N/A',
            'sprint': 'N/A',
            'sprint_team': 'N/A',
            'story_points': 'N/A',
            'fixed_version': 'N/A',
            'components': 'N/A',
            'repositories': 'N/A',
            'time_spent': 'N/A'
        }
        
        # Try to extract ticket key (e.g., PROJ-123)
        ticket_key_match = re.search(r'([A-Z]+-\d+)', jira_content, re.IGNORECASE)
        if ticket_key_match:
            info['ticket_key'] = ticket_key_match.group(1)
        
        # Try to extract title/summary
        title_patterns = [
            r'Title[:\s]+(.+)',
            r'Summary[:\s]+(.+)',
            r'Subject[:\s]+(.+)'
        ]
        for pattern in title_patterns:
            match = re.search(pattern, jira_content, re.IGNORECASE)
            if match:
                info['title'] = match.group(1).strip()
                break
        
        # Try to extract assignee
        assignee_patterns = [
            r'Assignee[:\s]+(.+)',
            r'Assigned to[:\s]+(.+)'
        ]
        for pattern in assignee_patterns:
            match = re.search(pattern, jira_content, re.IGNORECASE)
            if match:
                info['assignee'] = match.group(1).strip()
                break
        
        # Try to extract status
        status_patterns = [
            r'Status[:\s]+(.+)',
            r'State[:\s]+(.+)'
        ]
        for pattern in status_patterns:
            match = re.search(pattern, jira_content, re.IGNORECASE)
            if match:
                info['status'] = match.group(1).strip()
                break
        
        # Extract TODO items for sub-tasks
        todo_patterns = [
            r'TODO[:\s]+(.+)',
            r'- \[ \] (.+)',
            r'Sub-?task[:\s]+(.+)'
        ]
        
        subtasks = []
        for pattern in todo_patterns:
            matches = re.findall(pattern, jira_content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                subtasks.append({
                    'key': 'N/A',
                    'summary': match.strip(),
                    'assignees': 'N/A'
                })
        
        # Extract comments (lines that look like comments)
        comment_lines = []
        lines = jira_content.split('\n')
        for line in lines:
            line = line.strip()
            if line and not any(line.startswith(prefix) for prefix in ['Title:', 'Summary:', 'Assignee:', 'Status:', 'TODO:', 'DONE:']):
                if len(line) > 20 and not re.match(r'^[A-Z]+-\d+', line):
                    comment_lines.append(line)
        
        # Create markdown content
        markdown_content = f"""# JIRA Ticket Analysis

## Ticket Information
- **Key:** {info['ticket_key']}
- **Title:** {info['title']}
- **Parent Ticket:** {info['parent_ticket']}
- **Project:** {info['project']}
- **Type:** {info['issue_type']}
- **Status:** {info['status']}
- **Priority:** {info['priority']}
- **Reporter:** {info['reporter']}
- **Assignee:** {info['assignee']}
- **Sprint:** {info['sprint']}
- **Sprint Team:** {info['sprint_team']}
- **Story Points:** {info['story_points']}
- **Fixed Version:** {info['fixed_version']}
- **Components:** {info['components']}
- **Repositories:** {info['repositories']}
- **Time Spent:** {info['time_spent']}

## Detailed Description
The JIRA ticket contains information extracted from the provided content. This analysis provides a structured view of the available data.

## Sub-tasks
| Key | Summary | Assignees |
|-----|---------|-----------|"""

        for subtask in subtasks[:5]:  # Limit to 5 subtasks
            markdown_content += f"\n| {subtask['key']} | {subtask['summary']} | {subtask['assignees']} |"

        if not subtasks:
            markdown_content += "\n| N/A | No sub-tasks found | N/A |"

        markdown_content += "\n\n## Comments"
        
        if comment_lines:
            for comment in comment_lines[:5]:  # Limit to 5 comments
                markdown_content += f"\n- {comment}"
        else:
            markdown_content += "\n- No specific comments extracted"

        return {
            'success': True,
            'markdown_content': markdown_content,
            'raw_content': jira_content,
            'method': 'fallback'
        }


def test_openai_integration():
    """Test OpenAI integration"""
    updater = OpenAIDocumentationUpdater()
    
    print(f"OpenAI Available: {updater.is_available()}")
    
    # Test with sample content
    sample_jira = """
JIRA-123: Implement user authentication

TODO:
- Add login endpoint
- Implement JWT token validation
- Create user session management

COMPLETED:
- Database schema created
- User model implemented

Description:
This ticket involves implementing a complete user authentication system.
"""
    
    sample_asciidoc = """= API Documentation

== Overview

This document describes the API endpoints.

== Authentication

Currently no authentication is implemented.
"""
    
    if updater.is_available():
        # Test Jira analysis
        analysis = updater.analyze_jira_content(sample_jira)
        print("Jira Analysis:", analysis.get('analysis', 'No analysis'))
    else:
        print("OpenAI not available, testing fallback methods...")
        analysis = updater._fallback_jira_analysis(sample_jira, True)
        print("Fallback Analysis:", analysis.get('analysis', 'No analysis'))


if __name__ == "__main__":
    test_openai_integration()
