import fitz  # PyMuPDF
import numpy as np
from tika import parser

HEADER_THRESHOLD = 0.1  # 10% della pagina
FOOTER_THRESHOLD = 0.9  # 90% della pagina

def detect_headers_footers(pdf_path):
    # Aprire il file PDF
    document = fitz.open(pdf_path)
    page_rectangles = []

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        blocks = page.get_text("blocks")
        
        # Ordina i blocchi per la coordinata y (dal più alto al più basso)
        blocks = sorted(blocks, key=lambda b: b[1])
        
        header_blocks = []
        footer_blocks = []
        body_blocks = []
        
        page_height = page.rect.height
        header_threshold = page_height * HEADER_THRESHOLD
        footer_threshold = page_height * FOOTER_THRESHOLD
        
        for block in blocks:
            if block[1] < header_threshold:
                header_blocks.append(block)
            elif block[1] > footer_threshold:
                footer_blocks.append(block)
            else:
                body_blocks.append(block)
        
        # Calcola le coordinate del rettangolo del corpo del testo
        if body_blocks:
            min_y = min(block[1] for block in body_blocks)
            max_y = max(block[3] for block in body_blocks)
        else:
            min_y = header_threshold
            max_y = footer_threshold
        
        rect = fitz.Rect(0, min_y, page.rect.width, max_y)
        page_rectangles.append(rect)
        
        page.draw_rect(rect, color=(1, 0, 0), width=1)  # Rettangolo rosso con larghezza 1
        
        #Estrae un paragrafo dal corpo del testo
        #if body_blocks:
        #   paragraph = page.get_text("text", clip=rect)
        #   print(f"Pagina {page_num + 1} - Paragrafo estratto:\n{paragraph}\n")
    
    # Salva il nuovo file PDF
    output_path = "output_with_rectangles.pdf"
    document.save(output_path)
    document.close()
    
    return page_rectangles


# Esempio
pdf_path = "sample5.pdf"
rectangles = detect_headers_footers(pdf_path)
for i, rect in enumerate(rectangles):
    print(f"Pagina {i+1}: {rect}")