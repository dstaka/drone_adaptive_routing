#!/usr/bin/env python
"""Export the measured Japanese Markdown report with an embedded OFL font."""
import argparse
from pathlib import Path
import re
from xml.sax.saxutils import escape
from fontTools.ttLib import TTFont as FontToolsFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image as PillowImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='docs/report_ja.md')
    p.add_argument('--font', default='outputs/assets/NotoSansJP.ttf')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    static_font = out.with_suffix('.font.ttf')
    if out.exists() or static_font.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    font = FontToolsFont(a.font)
    if 'fvar' in font:
        instantiateVariableFont(font, {'wght': 400}, inplace=True)
    font.save(static_font)
    pdfmetrics.registerFont(TTFont('NotoJP', str(static_font)))
    body = ParagraphStyle('body', fontName='NotoJP', fontSize=9.5, leading=14.5,
                          wordWrap='CJK', spaceAfter=7, textColor=colors.HexColor('#203744'))
    heading = ParagraphStyle('heading', parent=body, fontSize=13, leading=18, spaceBefore=9,
                             textColor=colors.HexColor('#156e8c'))
    title = ParagraphStyle('title', parent=body, fontSize=20, leading=28, spaceAfter=16)
    story = []
    for line in Path(a.source).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak())
            continue
        match = re.match(r'!\[.*\]\((.+)\)', line)
        if match:
            image_path = Path(match.group(1))
            with PillowImage.open(image_path) as image:
                width, height = image.size
            story.extend([Image(str(image_path), width=480, height=480 * height / width), Spacer(1, 10)])
        elif line.startswith('# '):
            story.append(Paragraph(escape(line[2:]), title))
        elif line.startswith('## '):
            story.append(Paragraph(escape(line[3:]), heading))
        else:
            story.append(Paragraph(escape(line), body))
    def footer(canvas, doc):
        canvas.setFont('NotoJP', 8)
        canvas.setFillColor(colors.HexColor('#657c89'))
        canvas.drawString(36, 24, 'Adaptive Drone Inspection | 2026-10-13 | 実行結果に基づく報告')
        canvas.drawRightString(A4[0] - 36, 24, str(doc.page))
    doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=36, leftMargin=36,
                            topMargin=36, bottomMargin=42, title='適応型ドローン巡回点検', author='dstaka')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'Created {out}')


if __name__ == '__main__':
    main()
