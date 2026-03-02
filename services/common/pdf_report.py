import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def is_available() -> bool:
    try:
        from reportlab.lib.pagesizes import letter
        return True
    except ImportError:
        return False

class PDFReport:
    def __init__(self, title: str):
        self.title = title
        self.elements = []
        try:
            from reportlab.lib.styles import getSampleStyleSheet
            self.styles = getSampleStyleSheet()
        except ImportError:
            self.styles = None

    def add_header(self, text: str):
        if self.styles:
            from reportlab.platypus import Paragraph, Spacer
            self.elements.append(Paragraph(text, self.styles['Heading2']))
            self.elements.append(Spacer(1, 10))

    def add_key_value_section(self, title: str, data: Dict[str, Any]):
        if self.styles:
            from reportlab.platypus import Paragraph, Table, TableStyle
            from reportlab.lib import colors
            self.elements.append(Paragraph(title, self.styles['Heading3']))
            table_data = [[k, str(v)] for k, v in data.items()]
            t = Table(table_data, colWidths=[150, 300])
            t.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('FONTSIZE', (0,0), (-1,-1), 9),
            ]))
            self.elements.append(t)

    def add_table(self, headers: List[str], rows: List[List[Any]]):
        if self.styles:
            from reportlab.platypus import Table, TableStyle
            from reportlab.lib import colors
            table_data = [headers] + rows
            t = Table(table_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ]))
            self.elements.append(t)

    def build(self) -> Optional[bytes]:
        if not self.styles:
            return None
        import io
        from reportlab.platypus import SimpleDocTemplate
        from reportlab.lib.pagesizes import letter
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        doc.build(self.elements)
        buffer.seek(0)
        return buffer.getvalue()

class PDFReportGenerator:
    @staticmethod
    def generate_tear_sheet(metrics: Dict[str, Any], output_path: str = "APEX_Institutional_Tear_Sheet.pdf") -> bool:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
        except ImportError:
            logger.error("🚨 reportlab not installed. Fallback to Markdown.")
            # Simple fallback
            return False

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        elements.append(Paragraph("APEX ALGORITHMIC TRADING - INSTITUTIONAL TEAR SHEET", styles['Title']))
        elements.append(Spacer(1, 20))
        data = [["Metric", "Apex System Value"]]
        for key, val in metrics.items():
            data.append([key, str(val)])
        t = Table(data, colWidths=[200, 150])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        elements.append(t)
        doc.build(elements)
        logger.info(f"📄 PDF Tear Sheet successfully generated at: {output_path}")
        return True
