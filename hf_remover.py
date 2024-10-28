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
from sentence_transformers import SentenceTransformer
import os
import shutil


output_directory = 'chunks_output/'

HEADER_THRESHOLD = 0.1  # 10% della pagina
FOOTER_THRESHOLD = 0.9  # 90% della pagina

EXCLUDE_HEADERS = True
EXCLUDE_FOOTERS = True
model = SentenceTransformer('thenlper/gte-small')

def detect_headers_footers(pdf_path):
    document = fitz.open(pdf_path)
    page_rectangles = []
    cropped_document = fitz.open()
    text_with_pages = []
    document_body = []

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
        document_body.append({
            "text": body_text,
            "page": page_num + 1
        })

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

    json_output = {
        "metadata": metadata,
        "document_body": document_body
    }

    with open('json_output.json', 'w', encoding='utf-8') as file:
        json.dump(json_output, file, ensure_ascii=False, indent=4)

    # return json_output

    # return [{ "index": i, "page": chunk_page_numbers[i], "text": chunk } for i, chunk in enumerate(chunks)]


def save_to_json(chunks):
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory, exist_ok=True)
    for i, chunk in enumerate(chunks, start=1):
        file_path = os.path.join(output_directory, f'chunk_{i}.json')
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(chunk, file, ensure_ascii=False, indent=4)


def recursive_chunking_with_pages(text_with_pages, chunk_size=800, chunk_overlap=300):
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
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
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


def create_embeddings(data):
    for chunk in data:
        chunk_content = chunk['text']
        embedding_vector = model.encode(chunk_content).tolist()
        chunk['embedding_of_chunk'] = embedding_vector 

    save_to_json(data)


def main():
    pdf_path = "./examples/meditations.pdf"
    start_time = time.time()
    chunks = detect_headers_footers(pdf_path)
    end_time = time.time()
    # print(f"Tempo impiegato per l'estrazione: {end_time - start_time} secondi")
    # create_embeddings(chunks)
    

if __name__ == "__main__":
    main()