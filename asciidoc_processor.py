import re
from typing import Dict, List, Tuple, Optional
import difflib
from dataclasses import dataclass


@dataclass
class AsciiDocSection:
    """Represents a section in an AsciiDoc document"""
    level: int
    title: str
    content: str
    start_line: int
    end_line: int
    heading_marker: str


class AsciiDocProcessor:
    """Handles AsciiDoc specific processing and formatting"""
    
    def __init__(self):
        self.heading_patterns = {
            1: r'^=\s+(.+)$',
            2: r'^==\s+(.+)$', 
            3: r'^===\s+(.+)$',
            4: r'^====\s+(.+)$',
            5: r'^=====\s+(.+)$',
            6: r'^======\s+(.+)$'
        }
        
        # Compile regex patterns
        self.compiled_patterns = {
            level: re.compile(pattern, re.MULTILINE) 
            for level, pattern in self.heading_patterns.items()
        }
    
    def analyze_structure(self, content: str) -> Dict:
        """
        Analyze the structure of an AsciiDoc document
        
        Args:
            content (str): AsciiDoc content
            
        Returns:
            Dict: Analysis results including headings, sections, etc.
        """
        lines = content.split('\n')
        structure = {
            'headings': 0,
            'sections': [],
            'heading_levels': {},
            'total_lines': len(lines),
            'has_title': False
        }
        
        # Find all headings
        for level, pattern in self.compiled_patterns.items():
            matches = pattern.findall(content)
            if matches:
                structure['heading_levels'][level] = len(matches)
                structure['headings'] += len(matches)
                
                # Check for title (level 1)
                if level == 1 and matches:
                    structure['has_title'] = True
        
        # Extract sections
        sections = self.extract_sections(content)
        structure['sections'] = sections
        
        return structure
    
    def extract_sections(self, content: str) -> List[AsciiDocSection]:
        """
        Extract all sections from AsciiDoc content
        
        Args:
            content (str): AsciiDoc content
            
        Returns:
            List[AsciiDocSection]: List of document sections
        """
        lines = content.split('\n')
        sections = []
        current_section = None
        
        for line_num, line in enumerate(lines):
            # Check if line is a heading
            heading_match = self._parse_heading_line(line)
            
            if heading_match:
                # Save previous section if exists
                if current_section:
                    current_section.end_line = line_num - 1
                    sections.append(current_section)
                
                # Start new section
                level, title, marker = heading_match
                current_section = AsciiDocSection(
                    level=level,
                    title=title,
                    content="",
                    start_line=line_num,
                    end_line=line_num,
                    heading_marker=marker
                )
            elif current_section:
                # Add content to current section
                current_section.content += line + '\n'
        
        # Add final section
        if current_section:
            current_section.end_line = len(lines) - 1
            sections.append(current_section)
        
        return sections
    
    def _parse_heading_line(self, line: str) -> Optional[Tuple[int, str, str]]:
        """
        Parse a line to check if it's a heading
        
        Args:
            line (str): Line to check
            
        Returns:
            Optional[Tuple[int, str, str]]: (level, title, marker) or None
        """
        line = line.strip()
        
        for level, pattern in self.compiled_patterns.items():
            match = pattern.match(line)
            if match:
                title = match.group(1).strip()
                marker = '=' * level
                return (level, title, marker)
        
        return None
    
    def add_fastdoc_markers(self, original_content: str, updated_content: str) -> str:
        """
        Add FASTDOC markers to indicate changes in the updated content
        
        Args:
            original_content (str): Original AsciiDoc content
            updated_content (str): Updated AsciiDoc content
            
        Returns:
            str: Updated content with FASTDOC markers
        """
        # If content is identical, return original without markers
        if original_content.strip() == updated_content.strip():
            return updated_content
        
        # Split content into sections by headings
        original_sections = self._split_into_sections(original_content)
        updated_sections = self._split_into_sections(updated_content)
        
        result_lines = []
        
        for section in updated_sections:
            section_heading = section['heading']
            section_content = section['content']
            section_lines = section['lines']
            
            # Find corresponding section in original
            original_section = self._find_matching_section(section_heading, original_sections)
            
            if original_section is None:
                # This is a completely new section
                result_lines.append(f"// *FASTDOC* - New Section Added")
                result_lines.extend(section_lines)
                result_lines.append(f"// *FASTDOC* - End New Section")
            else:
                # Compare section content to see if it's actually different
                if self._sections_are_different(original_section['content'], section_content):
                    # Section has been modified
                    result_lines.append(f"// *FASTDOC* - Updated Section")
                    result_lines.extend(section_lines)
                    result_lines.append(f"// *FASTDOC* - End Update")
                else:
                    # Section is unchanged, add without markers
                    result_lines.extend(section_lines)
        
        return '\n'.join(result_lines)

    def _split_into_sections(self, content: str) -> List[Dict]:
        """Split content into sections based on headings"""
        lines = content.split('\n')
        sections = []
        current_section = {
            'heading': None,
            'content': [],
            'lines': []
        }
        
        for line in lines:
            heading_match = self._parse_heading_line(line)
            
            if heading_match:
                # Save previous section if it has content
                if current_section['heading'] or current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content']).strip()
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'heading': heading_match[1],  # heading title
                    'content': [],
                    'lines': [line]
                }
            else:
                current_section['content'].append(line)
                current_section['lines'].append(line)
        
        # Add final section
        if current_section['heading'] or current_section['content']:
            current_section['content'] = '\n'.join(current_section['content']).strip()
            sections.append(current_section)
        
        return sections
    
    def _find_matching_section(self, heading: str, sections: List[Dict]) -> Optional[Dict]:
        """Find a section with matching heading"""
        if not heading:
            return None
            
        for section in sections:
            if section['heading'] and section['heading'].lower().strip() == heading.lower().strip():
                return section
        return None
    
    def _sections_are_different(self, original_content: str, updated_content: str) -> bool:
        """
        Check if two section contents are meaningfully different
        
        Args:
            original_content (str): Original section content
            updated_content (str): Updated section content
            
        Returns:
            bool: True if sections are different enough to warrant FASTDOC markers
        """
        # Normalize content for comparison
        orig_normalized = self._normalize_content(original_content)
        updated_normalized = self._normalize_content(updated_content)
        
        # If content is identical after normalization, no change
        if orig_normalized == updated_normalized:
            return False
        
        # Check if there's substantial difference (more than just whitespace/formatting)
        orig_words = set(orig_normalized.split())
        updated_words = set(updated_normalized.split())
        
        # Calculate word-level difference
        added_words = updated_words - orig_words
        removed_words = orig_words - updated_words
        
        # Consider it different if there are new words added or significant changes
        significant_change_threshold = 3  # At least 3 new/changed words
        
        # Check for meaningful new content
        meaningful_additions = [word for word in added_words 
                              if len(word) > 2 and not word.isdigit()]
        
        return (len(meaningful_additions) >= significant_change_threshold or
                len(added_words) + len(removed_words) >= significant_change_threshold * 2)
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison"""
        if not content:
            return ""
        
        # Remove extra whitespace and normalize
        normalized = re.sub(r'\s+', ' ', content.strip())
        # Remove common markup that doesn't affect meaning
        normalized = re.sub(r'[*_`]', '', normalized)
        # Convert to lowercase for comparison
        normalized = normalized.lower()
        
        return normalized

    def render_for_display(self, content: str) -> str:
        """
        Render AsciiDoc content for display in Streamlit
        
        Args:
            content (str): AsciiDoc content
            
        Returns:
            str: HTML formatted content for display
        """
        import html
        
        # Convert AsciiDoc to HTML-like formatting for Streamlit
        lines = content.split('\n')
        html_lines = []
        
        for line in lines:
            # Escape HTML characters first to prevent issues
            safe_line = html.escape(line)
            
            # Handle headings
            heading_match = self._parse_heading_line(line)
            if heading_match:
                level, title, _ = heading_match
                # Ensure title is safe for HTML
                safe_title = html.escape(title)
                html_lines.append(f"<h{level}>{safe_title}</h{level}>")
                continue
            
            # Handle FASTDOC markers
            if "FASTDOC" in line:
                # Escape the line content to prevent HTML issues
                escaped_line = html.escape(line)
                if "New Section" in line:
                    html_lines.append(f'<div class="fastdoc-marker" style="background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 8px; margin: 8px 0; font-weight: bold; color: #1976D2;">🆕 {escaped_line}</div>')
                elif "Updated Section" in line:
                    html_lines.append(f'<div class="fastdoc-marker" style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 8px; margin: 8px 0; font-weight: bold; color: #856404;">✏️ {escaped_line}</div>')
                elif "End Update" in line:
                    html_lines.append(f'<div class="fastdoc-marker" style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 8px; margin: 8px 0; font-weight: bold; color: #155724;">✅ {escaped_line}</div>')
                else:
                    html_lines.append(f'<div class="fastdoc-marker" style="background-color: #f8f9fa; border-left: 4px solid #6c757d; padding: 8px; margin: 8px 0; font-weight: bold;">🏷️ {escaped_line}</div>')
                continue
            
            # Handle code blocks
            if line.strip().startswith('----'):
                if line.count('----') >= 2:  # End of code block
                    html_lines.append('</code></pre>')
                else:  # Start of code block
                    html_lines.append('<pre style="background-color: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto;"><code>')
                continue
            
            # Now work with the original line for pattern matching, but use escaped version for output
            original_line = line
            
            # Handle bold text
            safe_line = re.sub(r'\*\*(.+?)\*\*', lambda m: f'<strong>{html.escape(m.group(1))}</strong>', original_line)
            
            # Handle italic text  
            safe_line = re.sub(r'\*([^*]+?)\*', lambda m: f'<em>{html.escape(m.group(1))}</em>', safe_line)
            
            # Handle inline code
            safe_line = re.sub(r'`([^`]+?)`', lambda m: f'<code style="background-color: #f1f3f4; padding: 2px 4px; border-radius: 3px;">{html.escape(m.group(1))}</code>', safe_line)
            
            # Handle lists
            if original_line.strip().startswith('*'):
                safe_line = re.sub(r'^\s*\*\s*(.+)', lambda m: f'• {html.escape(m.group(1))}', original_line)
            elif original_line.strip().startswith('.'):
                safe_line = re.sub(r'^\s*\.\s*(.+)', lambda m: f'1. {html.escape(m.group(1))}', original_line)
            
            # Handle blockquotes (NOTE: admonition)
            if original_line.strip().startswith('NOTE:'):
                note_content = html.escape(original_line[5:].strip())
                safe_line = f'<div style="background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 12px; margin: 12px 0;"><strong>ℹ️ Note:</strong> {note_content}</div>'
            elif original_line.strip().startswith('WARNING:'):
                warning_content = html.escape(original_line[8:].strip())
                safe_line = f'<div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 12px 0;"><strong>⚠️ Warning:</strong> {warning_content}</div>'
            elif original_line.strip().startswith('TIP:'):
                tip_content = html.escape(original_line[4:].strip())
                safe_line = f'<div style="background-color: #d1ecf1; border-left: 4px solid #bee5eb; padding: 12px; margin: 12px 0;"><strong>💡 Tip:</strong> {tip_content}</div>'
            
            # Regular paragraphs
            if original_line.strip():
                # Only apply paragraph tags if it's not already formatted
                if not any(tag in safe_line for tag in ['<div', '<strong>', '<em>', '<code>', '<h']):
                    safe_line = f'<p>{html.escape(original_line)}</p>'
                html_lines.append(safe_line)
            else:
                html_lines.append('<br>')
        
        return '\n'.join(html_lines)
    
    def validate_asciidoc_syntax(self, content: str) -> Dict:
        """
        Validate AsciiDoc syntax and return issues
        
        Args:
            content (str): AsciiDoc content to validate
            
        Returns:
            Dict: Validation results
        """
        issues = []
        lines = content.split('\n')
        
        # Check heading sequence
        previous_level = 0
        for line_num, line in enumerate(lines):
            heading_match = self._parse_heading_line(line)
            if heading_match:
                level, title, _ = heading_match
                
                # Check if heading level jumps too much
                if level > previous_level + 1:
                    issues.append({
                        'line': line_num + 1,
                        'type': 'heading_sequence',
                        'message': f'Heading level jumps from {previous_level} to {level}',
                        'content': line
                    })
                
                previous_level = level
        
        # Check for common syntax issues
        for line_num, line in enumerate(lines):
            # Check for potential formatting issues
            if '**' in line and line.count('**') % 2 != 0:
                issues.append({
                    'line': line_num + 1,
                    'type': 'formatting',
                    'message': 'Unmatched bold markers (**)',
                    'content': line
                })
        
        return {
            'is_valid': len(issues) == 0,
            'issues_count': len(issues),
            'issues': issues
        }


def test_asciidoc_processor():
    """Test function for AsciiDoc processor"""
    sample_content = """= Main Title

This is the introduction paragraph.

== Section One

Content for section one with **bold text**.

=== Subsection

Some content here.

== Section Two  

More content with *italic text* and `inline code`.

----
code block content
here
----

NOTE: This is an important note.
"""

    processor = AsciiDocProcessor()
    
    # Test structure analysis
    structure = processor.analyze_structure(sample_content)
    print("Structure analysis:", structure)
    
    # Test section extraction
    sections = processor.extract_sections(sample_content)
    print(f"Found {len(sections)} sections:")
    for section in sections:
        print(f"  Level {section.level}: {section.title}")


if __name__ == "__main__":
    test_asciidoc_processor()
