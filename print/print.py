#!/usr/bin/env python3
"""
split_pdf_tiles.py: Split each page of a PDF into top and bottom halves,
and output each half as a separate A4 portrait page.

Usage:
    python split_pdf_tiles.py input.pdf output.pdf

Dependencies:
    pip install PyMuPDF
"""
import sys
import fitz  # PyMuPDF

def split_pdf(input_path: str, output_path: str) -> None:
    """
    開始PDFを読み込み、各ページを上下2分割して新規PDFに出力します。

    :param input_path: 入力PDFファイルパス
    :param output_path: 出力PDFファイルパス
    """
    doc = fitz.open(input_path)
    new_doc = fitz.open()

    for page in doc:
        rect = page.rect
        width, height = rect.width, rect.height
        half_h = height / 2
        # 上下2分割用クリップ範囲
        rects = [
            fitz.Rect(0, 0, width, half_h),         # 上半分
            fitz.Rect(0, half_h, width, height)     # 下半分
        ]
        # 各クリップ領域を新規ページとして追加
        for clip in rects:
            new_page = new_doc.new_page(width=width, height=half_h)
            new_page.show_pdf_page(new_page.rect, doc, page.number, clip=clip)

    new_doc.save(output_path)
    print(f"Saved split PDF to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_pdf_tiles.py input.pdf output.pdf")
        sys.exit(1)
    split_pdf(sys.argv[1], sys.argv[2])
