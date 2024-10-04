import fitz  # PyMuPDF
import json
import nltk
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
import numpy as np
import time
import re
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter



HEADER_THRESHOLD = 0.075  # 10% della pagina
FOOTER_THRESHOLD = 0.9  # 90% della pagina

EXCLUDE_HEADERS = True
EXCLUDE_FOOTERS = True

def detect_headers_footers(pdf_path):
    document = fitz.open(pdf_path)
    page_rectangles = []
    cropped_document = fitz.open()
    text_with_pages = []  # Store tuples of (text, page_number)

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: b[1])
        
        header_blocks = []
        footer_blocks = []
        body_blocks = []
        
        page_height = page.rect.height
        header_threshold = 0
        footer_threshold = page_height

        if EXCLUDE_HEADERS:
            header_threshold = page_height * HEADER_THRESHOLD
        
        if EXCLUDE_FOOTERS:
            footer_threshold = page_height * FOOTER_THRESHOLD
        
        for block in blocks:
            if block[1] < header_threshold:
                header_blocks.append(block)
            elif block[1] > footer_threshold:
                footer_blocks.append(block)
            else:
                body_blocks.append(block)

        if body_blocks:
            min_y = min(block[1] for block in body_blocks)
            max_y = max(block[3] for block in body_blocks)
        else:
            min_y = header_threshold
            max_y = footer_threshold
        
        rect = fitz.Rect(0, min_y, page.rect.width, max_y)
        page_rectangles.append(rect)
                
        cropped_page = page.set_cropbox(rect)
        cropped_document.insert_pdf(document, from_page=page_num, to_page=page_num)

        body_text = "\n".join([block[4] for block in body_blocks])
        text_with_pages.append((body_text, page_num + 1))

    cropped_output_path = "./output/cropped_output.pdf"
    cropped_document.save(cropped_output_path)
    cropped_document.close()

    chunks, chunk_page_numbers = recursive_chunking_with_pages(text_with_pages)

    metadata = {
        "title": document.metadata.get("title", "Unknown"),
        "author": document.metadata.get("author", "Unknown"),
        "num_pages": len(document),
        "num_chunks": len(chunks),
    }

    output_data = {
        "metadata": metadata,
        "chunks": [{ "page": chunk_page_numbers[i], "text": chunk, "index": i } for i, chunk in enumerate(chunks)]
    }

    json_output_path = "./output/output.json"
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)

    txt_output_path = "./output/output.txt"
    with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write("\n".join([text for text, _ in text_with_pages]))


def recursive_chunking_with_pages(text_with_pages, chunk_size=800, chunk_overlap=400):
    all_text = ""
    page_boundaries = []
    current_position = 0

    for text, page_number in text_with_pages:
        all_text += text
        page_boundaries.append((current_position, current_position + len(text), page_number))
        current_position += len(text)

    result = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
    ).create_documents([all_text])
    
    chunks = [doc.page_content for doc in result]
    chunk_page_numbers = []

    for chunk in chunks:
        chunk_start = all_text.find(chunk)
        chunk_end = chunk_start + len(chunk)

        for start, end, page_number in page_boundaries:
            if start <= chunk_start < end or start < chunk_end <= end:
                chunk_page_numbers.append(page_number)
                break

    return chunks, chunk_page_numbers


def fixed_size_chunking(text, chunk_size=100):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def split_text_into_chunks(text, chunk_size=300):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


def paragraph_level_chunking(text):
    chunks = text.split('\n\n')
    chunks = [chunk for chunk in chunks if chunk.strip() and not re.fullmatch(r'\d+', chunk.strip())]
    return chunks


# def recursive_chunking(text, chunk_size=1024):
#     if len(text) <= chunk_size:
#         return [text]
#     else:
#         mid = chunk_size
#         while mid > 0 and text[mid] != ' ':
#             mid -= 1
#         return [text[:mid]] + recursive_chunking(text[mid+1:], chunk_size)


def window_based_chunking(text, window_size=100, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        chunks.append(text[start:end])
        start += window_size - overlap
    return chunks


def tokenized_chunking(text, chunk_size=1000):
    nltk.download('punkt_tab')
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def sliding_window_chunking(text, window_size, step_size):
    chunks = []
    for i in range(0, len(text), step_size):
        chunk = text[i:i+window_size]
        chunks.append(chunk)
    return chunks


def main():
    pdf_path = "./examples/codice-civile.pdf"
    start_time = time.time()
    rectangles = detect_headers_footers(pdf_path)
    end_time = time.time()

    print(f"Tempo impiegato: {end_time - start_time} secondi")
    

if __name__ == "__main__":
    main()