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
        Add FASTDOC markers to indicate changes in the updated content using individual paragraph analysis
        
        Args:
            original_content (str): Original AsciiDoc content
            updated_content (str): Updated AsciiDoc content
            
        Returns:
            str: Updated content with FASTDOC markers around only the changed paragraphs
        """
        # If content is identical, return original without markers
        if original_content.strip() == updated_content.strip():
            return updated_content
        
        # Split content into paragraphs using double newlines as boundaries
        original_paragraphs = self._split_into_paragraphs(original_content)
        updated_paragraphs = self._split_into_paragraphs(updated_content)
        
        # Create a mapping of updated paragraphs to their change status
        paragraph_changes = []
        
        # Track which original paragraphs have been matched
        matched_original = set()
        
        # First pass: identify unchanged paragraphs
        for updated_para in updated_paragraphs:
            change_info = {'paragraph': updated_para, 'status': 'new', 'original_match': None}
            
            # Look for exact match in original (unchanged paragraph)
            for i, original_para in enumerate(original_paragraphs):
                if i not in matched_original and self._paragraphs_identical(original_para, updated_para):
                    change_info['status'] = 'unchanged'
                    change_info['original_match'] = i
                    matched_original.add(i)
                    break
            
            paragraph_changes.append(change_info)
        
        # Second pass: identify modifications for paragraphs still marked as 'new'
        for change_info in paragraph_changes:
            if change_info['status'] == 'new':
                updated_para = change_info['paragraph']
                
                # Look for best similarity match among unmatched originals
                best_match_idx = None
                best_similarity = 0
                
                for i, original_para in enumerate(original_paragraphs):
                    if i not in matched_original:
                        similarity = self._calculate_paragraph_similarity(original_para, updated_para)
                        if similarity > best_similarity and similarity > 0.7:  # Higher threshold to reduce false positives
                            best_similarity = similarity
                            best_match_idx = i
                
                if best_match_idx is not None:
                    # This is a modification
                    matched_original.add(best_match_idx)
                    original_para = original_paragraphs[best_match_idx]
                    
                    if self._is_substantial_paragraph(updated_para):
                        change_type = self._analyze_single_paragraph_change(original_para, updated_para)
                        if change_type == 'unchanged':
                            change_info['status'] = 'unchanged'
                        elif change_type == 'minor_edit':
                            change_info['status'] = 'minor_edit'
                        elif change_type == 'expansion':
                            change_info['status'] = 'expansion'
                        else:
                            change_info['status'] = 'modification'
                    else:
                        change_info['status'] = 'unchanged'
                    
                    change_info['original_match'] = best_match_idx
                # If no match found, it remains 'new'
        
        # Build the final result with markers only around changed paragraphs
        result_paragraphs = []
        
        for change_info in paragraph_changes:
            para = change_info['paragraph']
            status = change_info['status']
            
            if status == 'unchanged':
                # No markers for unchanged paragraphs
                result_paragraphs.append(para)
            elif status in ['minor_edit', 'expansion', 'modification']:
                result_paragraphs.append("// *FASTDOC* - Start Modification")
                result_paragraphs.append(para)
                result_paragraphs.append("// *FASTDOC* - End Modification")
            elif status == 'new':
                if self._is_substantial_paragraph(para):
                    result_paragraphs.append("// *FASTDOC* - Start Modification")
                    result_paragraphs.append(para)
                    result_paragraphs.append("// *FASTDOC* - End Modification")
                else:
                    result_paragraphs.append(para)
        
        # Add notice about deleted paragraphs at the end
        deleted_paragraphs = []
        for i, original_para in enumerate(original_paragraphs):
            if i not in matched_original and self._is_substantial_paragraph(original_para):
                deleted_paragraphs.append(original_para)
        
        if deleted_paragraphs:
            result_paragraphs.append(f"// *FASTDOC* - Content Removed ({len(deleted_paragraphs)} paragraph(s))")
        
        # Join paragraphs back together, preserving original paragraph spacing
        return self._join_paragraphs(result_paragraphs)
    
    def _paragraphs_identical(self, para1: str, para2: str) -> bool:
        """
        Check if two paragraphs are identical (after normalizing whitespace and minor formatting)
        
        Args:
            para1 (str): First paragraph
            para2 (str): Second paragraph
            
        Returns:
            bool: True if paragraphs are identical
        """
        if not para1.strip() and not para2.strip():
            return True
        
        # First check exact match after basic whitespace normalization
        norm1 = ' '.join(para1.split())
        norm2 = ' '.join(para2.split())
        
        if norm1 == norm2:
            return True
        
        # Check if they're functionally identical after normalizing common formatting variations
        normalized1 = self._deep_normalize_for_comparison(para1)
        normalized2 = self._deep_normalize_for_comparison(para2)
        
        return normalized1 == normalized2
    
    def _deep_normalize_for_comparison(self, text: str) -> str:
        """
        Deep normalization for detecting functional equivalence
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Deeply normalized text
        """
        if not text.strip():
            return ""
        
        # Normalize whitespace
        normalized = ' '.join(text.split())
        
        # Normalize common AsciiDoc formatting variations
        # Convert both *emphasis* and _emphasis_ to the same format (they're functionally equivalent)
        normalized = re.sub(r'\*([^*\s][^*]*?[^*\s])\*', r'EMPHASIS:\1:EMPHASIS', normalized)  # *word* -> EMPHASIS:word:EMPHASIS
        normalized = re.sub(r'_([^_\s][^_]*?[^_\s])_', r'EMPHASIS:\1:EMPHASIS', normalized)    # _word_ -> EMPHASIS:word:EMPHASIS
        normalized = re.sub(r'\*([^*\s])\*', r'EMPHASIS:\1:EMPHASIS', normalized)              # *a* -> EMPHASIS:a:EMPHASIS  
        normalized = re.sub(r'_([^_\s])_', r'EMPHASIS:\1:EMPHASIS', normalized)                # _a_ -> EMPHASIS:a:EMPHASIS
        
        # Normalize bullet point variations
        normalized = re.sub(r'^[\s]*[*_\-]\s+', '• ', normalized, flags=re.MULTILINE)
        
        # Normalize quotes
        normalized = normalized.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        
        # Remove extra whitespace around punctuation
        normalized = re.sub(r'\s+([,.;:!?])', r'\1', normalized)
        normalized = re.sub(r'([,.;:!?])\s+', r'\1 ', normalized)
        
        # Convert to lowercase for final comparison
        return normalized.lower().strip()
    
    def _calculate_paragraph_similarity(self, para1: str, para2: str) -> float:
        """
        Calculate similarity between two paragraphs with special handling for AsciiDoc formatting
        
        Args:
            para1 (str): First paragraph
            para2 (str): Second paragraph
            
        Returns:
            float: Similarity ratio between 0.0 and 1.0
        """
        if not para1.strip() or not para2.strip():
            return 0.0
        
        # Normalize for comparison - remove most formatting but preserve structure
        normalized1 = self._normalize_for_comparison(para1)
        normalized2 = self._normalize_for_comparison(para2)
        
        # Basic similarity
        basic_similarity = difflib.SequenceMatcher(None, normalized1, normalized2).ratio()
        
        # Check for list structure similarity
        list1_items = self._extract_list_items(para1)
        list2_items = self._extract_list_items(para2)
        
        if list1_items and list2_items:
            # Both are lists - compare the list items
            # If one list has all items from the other plus more, it's likely an expansion
            set1 = set(item.strip() for item in list1_items)
            set2 = set(item.strip() for item in list2_items)
            
            if set1.issubset(set2) or set2.issubset(set1):
                # One is a subset of the other - high similarity
                return max(0.6, basic_similarity)
            else:
                # Lists with some overlap
                common_items = len(set1.intersection(set2))
                total_items = len(set1.union(set2))
                if total_items > 0:
                    list_similarity = common_items / total_items
                    return max(basic_similarity, list_similarity * 0.8)
        
        return basic_similarity
    
    def _normalize_for_comparison(self, text: str) -> str:
        """
        Normalize text for comparison while preserving structure
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Normalized text
        """
        # Convert various bullet styles to a standard format
        normalized = re.sub(r'^\s*[*_\-]\s+', '• ', text, flags=re.MULTILINE)
        
        # Normalize bold/italic formatting
        normalized = re.sub(r'\*\*([^*]+)\*\*', r'_\1_', normalized)  # **bold** -> _italic_
        normalized = re.sub(r'\*([^*]+)\*', r'_\1_', normalized)       # *bold* -> _italic_
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip().lower()
    
    def _extract_list_items(self, text: str) -> List[str]:
        """
        Extract list items from text
        
        Args:
            text (str): Text to extract list items from
            
        Returns:
            List[str]: List of extracted items
        """
        lines = text.split('\n')
        items = []
        
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('*') or stripped.startswith('_') or 
                stripped.startswith('-') or stripped.startswith('.')):
                # Extract the content after the bullet
                content = re.sub(r'^[*_\-\.]\s*', '', stripped)
                if content:
                    items.append(content)
        
        return items
    
    def _analyze_single_paragraph_change(self, old_paragraph: str, new_paragraph: str) -> str:
        """
        Analyze the type of change between two individual paragraphs
        
        Args:
            old_paragraph (str): Original paragraph
            new_paragraph (str): Updated paragraph
            
        Returns:
            str: Type of change ('unchanged', 'minor_edit', 'substantial_modification', 'expansion')
        """
        if not old_paragraph.strip() and not new_paragraph.strip():
            return 'unchanged'
        
        if not old_paragraph.strip():
            return 'substantial_modification'  # New content
        
        if not new_paragraph.strip():
            return 'substantial_modification'  # Content removed
        
        # Calculate similarity ratio
        similarity = difflib.SequenceMatcher(None, old_paragraph, new_paragraph).ratio()
        
        # If very similar, check word-level changes
        if similarity > 0.9:
            # Use deep normalization to check for functional equivalence
            norm_old = self._deep_normalize_for_comparison(old_paragraph)
            norm_new = self._deep_normalize_for_comparison(new_paragraph)
            
            if norm_old == norm_new:
                return 'unchanged'
            
            old_words = set(old_paragraph.lower().split())
            new_words = set(new_paragraph.lower().split())
            
            added_words = new_words - old_words
            removed_words = old_words - new_words
            
            # More stringent minor edit threshold
            if len(added_words) <= 2 and len(removed_words) <= 2:
                return 'minor_edit' if (added_words or removed_words) else 'unchanged'
        
        # Check for expansion (significant length increase)
        old_length = len(old_paragraph.split())
        new_length = len(new_paragraph.split())
        
        if new_length > old_length * 1.5 and similarity > 0.5:
            return 'expansion'
        
        # Determine if change is substantial - be more conservative
        if similarity < 0.5:
            return 'substantial_modification'
        elif similarity > 0.85:
            # High similarity - double-check with deep normalization
            norm_old = self._deep_normalize_for_comparison(old_paragraph)
            norm_new = self._deep_normalize_for_comparison(new_paragraph)
            
            if norm_old == norm_new:
                return 'unchanged'
            else:
                return 'minor_edit'
        else:
            return 'minor_edit'
    
    def _paragraphs_similar(self, para1: str, para2: str, threshold: float = 0.7) -> bool:
        """
        Check if two paragraphs are similar enough to be considered the same
        
        Args:
            para1 (str): First paragraph
            para2 (str): Second paragraph
            threshold (float): Similarity threshold (0.0 to 1.0)
            
        Returns:
            bool: True if paragraphs are similar
        """
        if not para1.strip() or not para2.strip():
            return False
        
        similarity = difflib.SequenceMatcher(None, para1.lower(), para2.lower()).ratio()
        return similarity >= threshold
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """
        Split content into logical AsciiDoc blocks (not just by double newlines)
        
        Args:
            content (str): Content to split
            
        Returns:
            List[str]: List of logical blocks (headings, paragraphs, lists, code blocks, etc.)
        """
        if not content.strip():
            return []
        
        lines = content.split('\n')
        blocks = []
        current_block_lines = []
        current_block_type = None
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            
            # Detect block types
            is_heading = self._parse_heading_line(line) is not None
            is_list_item = (stripped_line.startswith('*') or 
                          stripped_line.startswith('_') or 
                          stripped_line.startswith('-') or
                          stripped_line.startswith('.'))
            is_code_block = stripped_line.startswith('----')
            is_admonition = stripped_line.startswith(('NOTE:', 'WARNING:', 'TIP:', 'IMPORTANT:', 'CAUTION:'))
            is_empty = not stripped_line
            is_attribute = stripped_line.startswith((':')) or stripped_line.startswith('ifndef::') or stripped_line.startswith('endif::')
            is_anchor = stripped_line.startswith('[[') and stripped_line.endswith(']]')
            
            # Determine what to do based on current and previous context
            if is_empty:
                # Empty line - might end current block or be part of it
                if current_block_type == 'list':
                    # For lists, empty line might be internal formatting or end of list
                    # Look ahead to see if next non-empty line is also a list item
                    next_non_empty_idx = i + 1
                    while next_non_empty_idx < len(lines) and not lines[next_non_empty_idx].strip():
                        next_non_empty_idx += 1
                    
                    if (next_non_empty_idx < len(lines) and 
                        (lines[next_non_empty_idx].strip().startswith('*') or 
                         lines[next_non_empty_idx].strip().startswith('_') or
                         lines[next_non_empty_idx].strip().startswith('-') or
                         lines[next_non_empty_idx].strip().startswith('.'))):
                        # Next item is also a list item - include empty line in current block
                        current_block_lines.append(line)
                    else:
                        # End of list
                        if current_block_lines:
                            blocks.append('\n'.join(current_block_lines).strip())
                            current_block_lines = []
                            current_block_type = None
                else:
                    # For other types, empty line ends the block
                    if current_block_lines:
                        blocks.append('\n'.join(current_block_lines).strip())
                        current_block_lines = []
                        current_block_type = None
                
            elif is_heading:
                # Heading always starts a new block
                if current_block_lines:
                    blocks.append('\n'.join(current_block_lines).strip())
                current_block_lines = [line]
                current_block_type = 'heading'
                
            elif is_list_item:
                if current_block_type != 'list':
                    # Starting a new list block
                    if current_block_lines:
                        blocks.append('\n'.join(current_block_lines).strip())
                    current_block_lines = [line]
                    current_block_type = 'list'
                else:
                    # Continuing current list
                    current_block_lines.append(line)
                    
            elif is_code_block:
                if current_block_lines:
                    blocks.append('\n'.join(current_block_lines).strip())
                    current_block_lines = []
                
                # Handle code block as a single unit
                code_block_lines = [line]
                i += 1
                while i < len(lines):
                    code_block_lines.append(lines[i])
                    if lines[i].strip().startswith('----'):
                        break
                    i += 1
                
                blocks.append('\n'.join(code_block_lines))
                current_block_lines = []
                current_block_type = None
                
            elif is_attribute or is_anchor:
                if current_block_lines:
                    blocks.append('\n'.join(current_block_lines).strip())
                blocks.append(line.strip())
                current_block_lines = []
                current_block_type = None
                
            elif is_admonition:
                if current_block_lines:
                    blocks.append('\n'.join(current_block_lines).strip())
                current_block_lines = [line]
                current_block_type = 'admonition'
                
            else:
                # Regular content line
                if current_block_type is None:
                    current_block_type = 'paragraph'
                current_block_lines.append(line)
            
            i += 1
        
        # Add final block
        if current_block_lines:
            blocks.append('\n'.join(current_block_lines).strip())
        
        return [block for block in blocks if block]
    
    def _join_paragraphs(self, paragraphs: List[str]) -> str:
        """
        Join paragraphs back together with proper spacing and visual separation
        
        Args:
            paragraphs (List[str]): List of paragraphs to join
            
        Returns:
            str: Joined content with proper paragraph spacing and marker separation
        """
        if not paragraphs:
            return ""
        
        result_parts = []
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph:
                continue
                
            current_is_marker = paragraph.strip().startswith('// *FASTDOC*')
            prev_is_marker = (i > 0 and paragraphs[i-1].strip().startswith('// *FASTDOC*'))
            next_is_marker = (i < len(paragraphs)-1 and paragraphs[i+1].strip().startswith('// *FASTDOC*'))
            
            # Add the paragraph
            result_parts.append(paragraph)
            
            # Determine spacing for next paragraph
            if i < len(paragraphs) - 1 and paragraphs[i + 1]:
                if current_is_marker or next_is_marker:
                    # Single newline between markers and content
                    result_parts.append('\n')
                else:
                    # Double newline between regular content paragraphs
                    # But add extra space if this content paragraph is followed by a marker
                    # to create visual separation
                    if (i + 1 < len(paragraphs) - 1 and 
                        paragraphs[i + 1].strip().startswith('// *FASTDOC*')):
                        # This unchanged paragraph is followed by a marker - add extra space
                        result_parts.append('\n\n')
                    else:
                        result_parts.append('\n\n')
        
        return ''.join(result_parts)
    
    def _is_substantial_paragraph(self, paragraph: str) -> bool:
        """
        Check if a paragraph contains substantial content worth marking
        
        Args:
            paragraph (str): Paragraph to check
            
        Returns:
            bool: True if paragraph has substantial content
        """
        if not paragraph.strip():
            return False
        
        # Remove markup and count meaningful words
        cleaned = re.sub(r'[=*_`\[\](){}]', '', paragraph)
        words = [word for word in cleaned.split() if len(word) > 2 and not word.isdigit()]
        
        # Consider substantial if has meaningful content
        return (len(words) >= 3 or                    # At least 3 meaningful words
                self._parse_heading_line(paragraph) or  # Is a heading
                '----' in paragraph or                   # Code block
                paragraph.strip().startswith(('NOTE:', 'WARNING:', 'TIP:')))  # Admonition

    
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
                if "Start Modification" in line:
                    html_lines.append(f'<div class="fastdoc-marker" style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 8px; margin: 8px 0; font-weight: bold; color: #856404;">[MOD] {escaped_line}</div>')
                elif "End Modification" in line:
                    html_lines.append(f'<div class="fastdoc-marker" style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 8px; margin: 8px 0; font-weight: bold; color: #155724;">[END] {escaped_line}</div>')
                elif "Content Removed" in line:
                    html_lines.append(f'<div class="fastdoc-marker" style="background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 8px; margin: 8px 0; font-weight: bold; color: #721c24;">[DEL] {escaped_line}</div>')
                else:
                    html_lines.append(f'<div class="fastdoc-marker" style="background-color: #f8f9fa; border-left: 4px solid #6c757d; padding: 8px; margin: 8px 0; font-weight: bold;">[TAG] {escaped_line}</div>')
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
                safe_line = f'<div style="background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 12px; margin: 12px 0;"><strong>Note:</strong> {note_content}</div>'
            elif original_line.strip().startswith('WARNING:'):
                warning_content = html.escape(original_line[8:].strip())
                safe_line = f'<div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 12px 0;"><strong>Warning:</strong> {warning_content}</div>'
            elif original_line.strip().startswith('TIP:'):
                tip_content = html.escape(original_line[4:].strip())
                safe_line = f'<div style="background-color: #d1ecf1; border-left: 4px solid #bee5eb; padding: 12px; margin: 12px 0;"><strong>Tip:</strong> {tip_content}</div>'
            
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
    
    # Test the new paragraph-level FASTDOC marker functionality
    print("\n=== Testing Paragraph-Level FASTDOC Markers ===")
    
    # Create a modified version to test markers
    modified_content = """= Main Title

This is the introduction paragraph with additional details about the system.

== Section One

Content for section one with **bold text** and new information about implementation.

This is a completely new paragraph that was added.

=== Subsection

Some content here that has been updated with more details.

== Section Two  

More content with *italic text* and `inline code`.

== New Section

This is an entirely new section that was added to the documentation.

----
code block content
here
with new lines
----

NOTE: This is an important note.

WARNING: This is a new warning that was added.
"""

    # Test FASTDOC markers
    marked_content = processor.add_fastdoc_markers(sample_content, modified_content)
    
    print("Content with FASTDOC markers:")
    print("=" * 50)
    print(marked_content)
    print("=" * 50)
    
    # Count markers
    marker_count = marked_content.count("*FASTDOC*")
    print(f"Total FASTDOC markers added: {marker_count}")


def test_paragraph_detection():
    """Test the new individual paragraph detection and marking logic"""
    processor = AsciiDocProcessor()
    
    original = """= Title

First paragraph that remains unchanged.

Second paragraph with original content.

Third paragraph to be removed.

== Section

Section content here.

Another paragraph in the section."""

    updated = """= Title

First paragraph that remains unchanged.

Second paragraph with original content that has been modified with additional details.

== Section

Section content here with updates and new information.

Another paragraph in the section.

This is a completely new paragraph added to the section.

== New Section

Brand new section with new content."""

    print("=== Individual Paragraph Marking Test ===")
    
    # Test paragraph splitting
    orig_paragraphs = processor._split_into_paragraphs(original)
    updated_paragraphs = processor._split_into_paragraphs(updated)
    
    print(f"Original paragraphs: {len(orig_paragraphs)}")
    for i, p in enumerate(orig_paragraphs):
        print(f"  {i+1}: {p[:60]}...")
    
    print(f"\nUpdated paragraphs: {len(updated_paragraphs)}")
    for i, p in enumerate(updated_paragraphs):
        print(f"  {i+1}: {p[:60]}...")
    
    # Test individual paragraph change analysis
    print("\n=== Individual Paragraph Change Analysis ===")
    
    # Test some specific paragraph comparisons
    test_cases = [
        ("First paragraph that remains unchanged.", "First paragraph that remains unchanged."),
        ("Second paragraph with original content.", "Second paragraph with original content that has been modified with additional details."),
        ("Section content here.", "Section content here with updates and new information."),
    ]
    
    for old_para, new_para in test_cases:
        change_type = processor._analyze_single_paragraph_change(old_para, new_para)
        print(f"Change type: {change_type}")
        print(f"  Old: {old_para[:50]}...")
        print(f"  New: {new_para[:50]}...")
        print()
    
    # Test the full marking system
    print("=== Result with Individual Paragraph Markers ===")
    result = processor.add_fastdoc_markers(original, updated)
    print(result)
    
    # Count different types of markers
    marker_types = {
        'Start Modification': result.count('Start Modification'),
        'End Modification': result.count('End Modification'),
        'Content Removed': result.count('Content Removed'),
    }
    
    print(f"\n=== Marker Statistics ===")
    for marker_type, count in marker_types.items():
        if count > 0:
            print(f"{marker_type}: {count}")


def test_precise_paragraph_marking():
    """Test that only changed paragraphs get marked, not entire sections"""
    processor = AsciiDocProcessor()
    
    original = """= Documentation

This paragraph stays the same.

This paragraph will change.

This paragraph also stays the same.

== Section

Unchanged section paragraph.

Section paragraph to modify.

Another unchanged paragraph."""

    updated = """= Documentation

This paragraph stays the same.

This paragraph will change and now has additional content added to it.

This paragraph also stays the same.

== Section

Unchanged section paragraph.

Section paragraph to modify with new details and information.

Another unchanged paragraph.

New paragraph added to the section."""

    print("=== Precise Paragraph Marking Test ===")
    print("This test verifies that only the specific paragraphs that changed get markers.")
    print()
    
    result = processor.add_fastdoc_markers(original, updated)
    
    print("Result:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    
    # Verify that unchanged paragraphs don't have markers
    lines = result.split('\n')
    unchanged_paragraphs = [
        "This paragraph stays the same.",
        "This paragraph also stays the same.",
        "Unchanged section paragraph.",
        "Another unchanged paragraph."
    ]
    
    print("\n=== Verification ===")
    print("Checking that unchanged paragraphs are NOT directly marked:")
    print("(Being adjacent to markers for other paragraphs is expected and OK)")
    
    for unchanged in unchanged_paragraphs:
        # Find the line with this paragraph
        paragraph_line_index = None
        for i, line in enumerate(lines):
            if line.strip() == unchanged.strip():
                paragraph_line_index = i
                break
        
        if paragraph_line_index is not None:
            # Check if this paragraph is DIRECTLY marked (has start marker before AND end marker after)
            has_start_marker_before = (paragraph_line_index > 0 and 
                                     "*FASTDOC*" in lines[paragraph_line_index-1] and
                                     "Start Modification" in lines[paragraph_line_index-1])
            
            has_end_marker_after = (paragraph_line_index < len(lines)-1 and 
                                   "*FASTDOC*" in lines[paragraph_line_index+1] and
                                   "End Modification" in lines[paragraph_line_index+1])
            
            is_directly_marked = has_start_marker_before and has_end_marker_after
            
            if is_directly_marked:
                print(f"  INCORRECT: '{unchanged[:30]}...' is directly marked")
                print(f"    Start marker: {lines[paragraph_line_index-1].strip()}")
                print(f"    End marker: {lines[paragraph_line_index+1].strip()}")
            else:
                print(f"  CORRECT: '{unchanged[:30]}...' is not directly marked")
                
                # Show adjacent markers for context (this is expected and OK)
                adjacent_markers = []
                if (paragraph_line_index > 0 and "*FASTDOC*" in lines[paragraph_line_index-1]):
                    adjacent_markers.append(f"Before: {lines[paragraph_line_index-1].strip()}")
                if (paragraph_line_index < len(lines)-1 and "*FASTDOC*" in lines[paragraph_line_index+1]):
                    adjacent_markers.append(f"After: {lines[paragraph_line_index+1].strip()}")
                
                if adjacent_markers:
                    print(f"    Adjacent markers (OK): {' | '.join(adjacent_markers)}")
        else:
            print(f"  WARNING: Could not find paragraph: '{unchanged[:30]}...'")
            
    print("\nChecking that changed paragraphs ARE directly marked:")
    changed_paragraphs = [
        "This paragraph will change and now has additional content added to it.",
        "Section paragraph to modify with new details and information.",
        "New paragraph added to the section."
    ]
    
    for changed in changed_paragraphs:
        paragraph_line_index = None
        for i, line in enumerate(lines):
            if changed in line.strip():
                paragraph_line_index = i
                break
        
        if paragraph_line_index is not None:
            has_start_marker_before = (paragraph_line_index > 0 and 
                                     "*FASTDOC*" in lines[paragraph_line_index-1] and
                                     "Start Modification" in lines[paragraph_line_index-1])
            
            has_end_marker_after = (paragraph_line_index < len(lines)-1 and 
                                   "*FASTDOC*" in lines[paragraph_line_index+1] and
                                   "End Modification" in lines[paragraph_line_index+1])
            
            is_directly_marked = has_start_marker_before and has_end_marker_after
            
            if is_directly_marked:
                print(f"  CORRECT: '{changed[:30]}...' is directly marked")
            else:
                print(f"  INCORRECT: '{changed[:30]}...' is not directly marked")
                if paragraph_line_index > 0:
                    print(f"    Before: {lines[paragraph_line_index-1]}")
                if paragraph_line_index < len(lines)-1:
                    print(f"    After: {lines[paragraph_line_index+1]}")
        else:
            print(f"  WARNING: Could not find paragraph: '{changed[:30]}...'")


def test_false_positive_prevention():
    """Test that identical content doesn't get marked with FASTDOC markers"""
    processor = AsciiDocProcessor()
    
    # Test case 1: Completely identical content
    identical_content = """Child Strategies are enabled for parent strategies that support SOR child strategies.
For example a VWAP parent strategy can have a Quod Financial Lit SOR child strategy."""
    
    result = processor.add_fastdoc_markers(identical_content, identical_content)
    
    print("=== False Positive Prevention Test ===")
    print("Testing identical content marking...")
    
    if "*FASTDOC*" in result:
        print("FAIL: Identical content was marked with FASTDOC markers")
        print("Result:", result)
    else:
        print("PASS: Identical content was not marked")
    
    # Test case 2: Minor formatting differences that should be ignored
    original = "Child Strategies are enabled for parent strategies that support SOR *child* strategies."
    updated = "Child Strategies are enabled for parent strategies that support SOR _child_ strategies."
    
    result2 = processor.add_fastdoc_markers(original, updated)
    
    print("\nTesting minor formatting differences...")
    if "*FASTDOC*" in result2:
        print("FAIL: Minor formatting differences triggered FASTDOC markers")
        print("Result:", result2)
    else:
        print("PASS: Minor formatting differences were ignored")
    
    # Test case 3: Real change should still be detected
    original3 = "Child Strategies are enabled for parent strategies."
    updated3 = "Child Strategies are enabled for parent strategies that support SOR child strategies with new functionality."
    
    result3 = processor.add_fastdoc_markers(original3, updated3)
    
    print("\nTesting real content changes...")
    if "*FASTDOC*" in result3:
        print("PASS: Real content changes were detected and marked")
    else:
        print("FAIL: Real content changes were not detected")
        print("Result:", result3)


def test_user_example():
    """Test the specific user example with AsciiDoc list modifications"""
    processor = AsciiDocProcessor()
    
    original = """ifndef::imagesDir[]
:imagesDir: http://download-pa3.quodfinancial.com/graphics/docimages/TradingFE
endif::[]

[[frontendribbontkt-beta-feribbontkt]]
=== Ticket (Beta)

Create *Buy (Beta)* or *Sell (Beta)* orders using the Equity Trading *Order ticket*.

This section covers:

* <<frontendribbontkt-beta-feribbontkt-buysell>>
* <<frontendribbontkt-beta-feribbontkt-advanced>>
* <<frontendribbontkt-beta-feribbontkt-depth>>
* <<frontendribbontkt-beta-feribbontkt-order-ticket-confirmation>>
* <<frontendribbontkt-beta-feribbontkt-customize>>

The Order Ticket (Beta) has several workspace properties:

* It can be docked anywhere in a user's workspace (see <<common-workspace-docking>>)
* It can be piloted, establishing link between multiple panels or instruments where the content in one panel/instrument (called the 'piloted' panel)
is driven from a selection in the originating panel/instrument (the 'out-piloting' panel). See <<common-commonactions-piloting>>
* Multiple Order Tickets can be opened simultaneously
* The layout can be <<frontendribbontkt-beta-feribbontkt-customize,customized>>
* The layout can be <<common-workspace-cloning,cloned>>
* Multiple layouts can be <<common-workspace-layouts,saved and reopened>>"""

    updated = """ifndef::imagesDir[]
:imagesDir: http://download-pa3.quodfinancial.com/graphics/docimages/TradingFE
endif::[]

[[frontendribbontkt-beta-feribbontkt]]
=== Ticket (Beta)

Create _Buy (Beta)_ or _Sell (Beta)_ orders using the Equity Trading _Order ticket_.

This section covers:

_ <<frontendribbontkt-beta-feribbontkt-buysell>>
_ <<frontendribbontkt-beta-feribbontkt-advanced>>
_ <<frontendribbontkt-beta-feribbontkt-depth>>
_ <<frontendribbontkt-beta-feribbontkt-order-ticket-confirmation>>
_ <<frontendribbontkt-beta-feribbontkt-customize>>

The Order Ticket (Beta) has several workspace properties:

_ It can be docked anywhere in a user's workspace (see <<common-workspace-docking>>)
_ It can be piloted, establishing a link between multiple panels or instruments where the content in one panel/instrument (called the 'piloted' panel) is driven from a selection in the originating panel/instrument (the 'out-piloting' panel). See <<common-commonactions-piloting>>
_ Multiple Order Tickets can be opened simultaneously
_ The layout can be <<frontendribbontkt-beta-feribbontkt-customize,customized>>
_ The layout can be <<common-workspace-cloning,cloned>>
_ Multiple layouts can be <<common-workspace-layouts,saved and reopened>>
_ A new tab group named "Trigger" has been added, which includes fields for order customization (see <<frontendribbontkt-beta-feribbontkt-trigger>>)."""

    print("=== User's Specific Example Test ===")
    print("Testing AsciiDoc list modifications and formatting changes")
    print()
    
    # Test the block splitting
    orig_blocks = processor._split_into_paragraphs(original)
    updated_blocks = processor._split_into_paragraphs(updated)
    
    print(f"Original blocks: {len(orig_blocks)}")
    for i, block in enumerate(orig_blocks):
        print(f"  {i+1}: {block[:60].replace(chr(10), ' ')}...")
    
    print(f"\nUpdated blocks: {len(updated_blocks)}")  
    for i, block in enumerate(updated_blocks):
        print(f"  {i+1}: {block[:60].replace(chr(10), ' ')}...")
    
    # Test the change detection
    result = processor.add_fastdoc_markers(original, updated)
    
    print("\n=== Result with FASTDOC Markers ===")
    print(result)
    
    # Verify the expected markers are present
    expected_markers = [
        "// *FASTDOC* - Start Modification",
        "// *FASTDOC* - End Modification"
    ]
    
    marker_count = 0
    for marker in expected_markers:
        count = result.count(marker)
        marker_count += count
        print(f"\n'{marker}': {count} occurrences")
    
    print(f"\nTotal FASTDOC markers: {marker_count}")
    
    # Check if the new list item is included
    if "Trigger" in result and "order customization" in result:
        print("New list item with 'Trigger' is present")
    else:
        print("New list item with 'Trigger' is missing")
    
    # Check if formatting changes are handled
    if "_Buy (Beta)_" in result and "_Order ticket_" in result:
        print("Formatting changes from * to _ are present")
    else:
        print("Formatting changes are missing")


if __name__ == "__main__":
    test_false_positive_prevention()
    print("\n" + "="*80 + "\n")
    test_asciidoc_processor()
    print("\n" + "="*80 + "\n")
    test_paragraph_detection() 
    print("\n" + "="*80 + "\n")
    test_precise_paragraph_marking()
    print("\n" + "="*80 + "\n")
    test_user_example()
