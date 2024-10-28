import re
from collections import defaultdict
from typing import List, Tuple, Set


def detect_page_boundaries(text: str) -> List[int]:
    lines = text.split('\n')
    boundaries = []
    
    blank_count = 0
    for i, line in enumerate(lines):
        if line.strip() == '':
            blank_count += 1
        else:
            if blank_count >= 2:
                boundaries.append(i - blank_count)
            blank_count = 0
    
    return boundaries

def find_repeated_patterns(lines: List[str], window_size: int = 5) -> Set[Tuple[str, ...]]:
    patterns = defaultdict(int)
    
    line_tuples = [tuple(lines[i:i+window_size]) for i in range(len(lines)-window_size+1)]
    
    for pattern in line_tuples:
        patterns[pattern] += 1
    
    return {pattern for pattern, count in patterns.items() if count > 1}

def remove_headers_footers(filename: str, output_filename: str = None):
    if output_filename is None:
        output_filename = 'cleaned_' + filename
    
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    lines = text.split('\n')
    boundaries = detect_page_boundaries(text)
    
    if not boundaries:
        print("No clear page boundaries detected.")
        return
    
    header_window = 5 
    footer_window = 5
    
    headers = []
    footers = []
    
    for i in range(len(boundaries)-1):
        page_start = boundaries[i]
        page_end = boundaries[i+1]
        
        # Skip if page is too short
        if page_end - page_start < header_window + footer_window:
            continue
            
        headers.append(lines[page_start:page_start+header_window])
        footers.append(lines[page_end-footer_window:page_end])
    
    # Find repeated patterns
    header_patterns = find_repeated_patterns([line for page in headers for line in page])
    footer_patterns = find_repeated_patterns([line for page in footers for line in page])
    
    cleaned_lines = []
    current_line = 0
    
    while current_line < len(lines):
        # Check if current position starts a header pattern
        skip_header = False
        for pattern in header_patterns:
            pattern_lines = list(pattern)
            if current_line + len(pattern_lines) <= len(lines):
                if lines[current_line:current_line+len(pattern_lines)] == pattern_lines:
                    current_line += len(pattern_lines)
                    skip_header = True
                    break
        
        if not skip_header:
            # Check if current position starts a footer pattern
            skip_footer = False
            for pattern in footer_patterns:
                pattern_lines = list(pattern)
                if current_line + len(pattern_lines) <= len(lines):
                    if lines[current_line:current_line+len(pattern_lines)] == pattern_lines:
                        current_line += len(pattern_lines)
                        skip_footer = True
                        break
            
            if not skip_footer:
                cleaned_lines.append(lines[current_line])
                current_line += 1
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
    
    print(f"Processed file saved as: {output_filename}")

if __name__ == "__main__":
    input_file = "./output/output.txt"
    output_file = "./output/output_cleaned.txt"
    
    remove_headers_footers(input_file, output_file)