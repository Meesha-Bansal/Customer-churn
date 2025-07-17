# utils/report_generator.py

import os

def generate_pdf_report(pdf_bytes, output_path):
    """
    Save uploaded PDF bytes to the given file path.
    :param pdf_bytes: Raw PDF file bytes
    :param output_path: Full path to save the PDF
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(pdf_bytes)
