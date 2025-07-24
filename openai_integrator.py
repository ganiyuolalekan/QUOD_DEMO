import os
from typing import Dict, List, Optional
import difflib
import re
from openai import OpenAI
import streamlit as st

from dotenv import load_dotenv

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
            st.error(f"❌ Error initializing OpenAI client: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if OpenAI client is available"""
        return self.client is not None
    
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
            st.error(f"❌ Error analyzing Jira content with OpenAI: {str(e)}")
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
    
    def update_documentation(self, jira_content: str, asciidoc_content: str, config: Dict) -> str:
        """
        Update AsciiDoc documentation based on Jira content
        
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
            # Generate diff prompt
            diff_lines = self._generate_diff(asciidoc_content, asciidoc_content)  # Initial empty diff
            
            system_prompt = """You are an expert in AsciiDoc documentation automation and change management.

CRITICAL REQUIREMENTS:
1. You MUST make actual, meaningful changes to the documentation content that includes updating the documentation with new information confirmed from Jira ticket content. 
2. The confirmed changes (which should be re-written text) should be marked as "MODIFIED", and the changes that well included should be marked as "ADDED".
3. All updated sections should be different from the section on the original document, using the jira input as context to update/modify that section
4. All sections that will receive FASTDOC markers must contain actual changes, not just copies

Your task:
1. Analyze the Jira ticket for specific information that needs to be integrated/added/modified in the documentation
2. Identify existing sections that has been updated based on Jira content
3. Create new sections only if the Jira ticket introduces completely new functionality
4. For UPDATED sections: Modify the actual content by:
   - Adding new implementation details from Jira (ADDED)
   - Updating descriptions with new requirements (MODIFIED)
   - Adding new parameters, methods, or features mentioned in Jira (ADDED)
   - Incorporating TODO items as future enhancements (ADDED)
   - Including completed work as current functionality (ADDED)

CHANGE EXAMPLES:
- If Jira mentions "added JWT authentication", update auth sections to include JWT details
- If Jira lists new API endpoints, add them to the API documentation
- If Jira describes bug fixes, update the affected sections with corrected information
- If Jira includes new configuration options, add them to configuration sections

FORMAT REQUIREMENTS:
- Use proper AsciiDoc heading syntax (=, ==, ===, etc.)
- Maintain existing document structure, only parts of the document that needs change or a new section should be amended
- Preserve all original content that is still relevant
- DO NOT add FASTDOC markers (these are added later)

VALIDATION:
Before finishing, ensure every section you modified/added contains information that was NOT in the original document."""
            
            user_prompt = f"""JIRA TICKET INFORMATION TO INTEGRATE:
{jira_content}

CURRENT ASCIIDOC DOCUMENTATION:
{asciidoc_content}

PROCESSING CONFIGURATION:
- Mode: {config.get('processing_mode', 'Update Existing Sections')}
- Include TODOs: {config.get('include_todos', True)}
- Preserve Formatting: {config.get('preserve_formatting', True)}

INSTRUCTIONS:
1. Carefully read the Jira ticket to understand what changes need to be made
2. Identify which sections of the current documentation need updates based on the Jira content
3. Make actual changes to those sections by incorporating Jira information
4. Ensure that updated sections contain new information not present in the original document
5. Return the complete updated AsciiDoc document

IMPORTANT: Do not simply copy existing sections. Every section that gets updated must contain actual changes derived from the Jira ticket content."""
            
            # Store prompts for display
            self.last_system_prompt = system_prompt
            self.last_user_prompt = user_prompt
            
            response = self.client.chat.completions.create(
                model=config.get('ai_model', 'gpt-4o'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
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
            st.error(f"❌ Error updating documentation with OpenAI: {str(e)}")
            return self._fallback_documentation_update(jira_content, asciidoc_content, config)
    
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
        """Reprocess documentation with custom prompts"""
        if not self.is_available():
            st.error("❌ OpenAI client not available for custom prompt processing")
            return ""
        
        try:
            # Store the custom prompts
            self.last_system_prompt = system_prompt
            self.last_user_prompt = user_prompt
            
            response = self.client.chat.completions.create(
                model=config.get('ai_model', 'gpt-4o'),
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
