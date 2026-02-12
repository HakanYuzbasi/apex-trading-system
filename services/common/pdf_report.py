"""
services/common/pdf_report.py - PDF report generation for SaaS features.

Uses reportlab to generate professional trading analysis PDF reports.
Falls back gracefully when reportlab is not installed.
"""

import io
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REPORTLAB_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _REPORTLAB_AVAILABLE = True
except ImportError:
    logger.warning("reportlab not installed - PDF generation disabled")


def is_available() -> bool:
    """Check if PDF generation is available."""
    return _REPORTLAB_AVAILABLE


class PDFReport:
    """Builder for structured PDF reports.

    Usage::

        report = PDFReport(title="Backtest Validation Report")
        report.add_header("Executive Summary")
        report.add_paragraph("The strategy shows a Sharpe of 1.8 ...")
        report.add_table(
            headers=["Metric", "Value"],
            rows=[["Sharpe", "1.8"], ["MaxDD", "-12%"]],
        )
        pdf_bytes = report.build()
    """

    def __init__(self, title: str = "Apex Trading Report", page_size=None):
        if not _REPORTLAB_AVAILABLE:
            raise RuntimeError("reportlab is not installed. Run: pip install reportlab")

        self._title = title
        self._page_size = page_size or letter
        self._elements: list = []
        self._styles = getSampleStyleSheet()

        # Custom styles
        self._styles.add(ParagraphStyle(
            "ReportTitle",
            parent=self._styles["Heading1"],
            fontSize=22,
            spaceAfter=20,
            textColor=colors.HexColor("#1a1a2e"),
        ))
        self._styles.add(ParagraphStyle(
            "SectionHeader",
            parent=self._styles["Heading2"],
            fontSize=14,
            spaceBefore=16,
            spaceAfter=8,
            textColor=colors.HexColor("#16213e"),
        ))
        self._styles.add(ParagraphStyle(
            "BodyText2",
            parent=self._styles["BodyText"],
            fontSize=10,
            leading=14,
        ))

        # Title page
        self._elements.append(Spacer(1, 1 * inch))
        self._elements.append(Paragraph(title, self._styles["ReportTitle"]))
        self._elements.append(Paragraph(
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            self._styles["BodyText2"],
        ))
        self._elements.append(Spacer(1, 0.3 * inch))

    def add_header(self, text: str) -> "PDFReport":
        """Add a section header."""
        self._elements.append(Paragraph(text, self._styles["SectionHeader"]))
        return self

    def add_paragraph(self, text: str) -> "PDFReport":
        """Add body text."""
        self._elements.append(Paragraph(text, self._styles["BodyText2"]))
        self._elements.append(Spacer(1, 6))
        return self

    def add_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        col_widths: Optional[List[float]] = None,
    ) -> "PDFReport":
        """Add a styled data table."""
        data = [headers] + rows
        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("TOPPADDING", (0, 1), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ]))
        self._elements.append(Spacer(1, 8))
        self._elements.append(table)
        self._elements.append(Spacer(1, 8))
        return self

    def add_key_value_section(
        self, title: str, data: Dict[str, Any]
    ) -> "PDFReport":
        """Add a section with key-value pairs as a two-column table."""
        self.add_header(title)
        rows = [[str(k), str(v)] for k, v in data.items()]
        self.add_table(headers=["Metric", "Value"], rows=rows)
        return self

    def add_page_break(self) -> "PDFReport":
        """Insert a page break."""
        self._elements.append(PageBreak())
        return self

    def add_spacer(self, height_inches: float = 0.3) -> "PDFReport":
        """Add vertical space."""
        self._elements.append(Spacer(1, height_inches * inch))
        return self

    def build(self) -> bytes:
        """Render the report and return PDF as bytes."""
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=self._page_size,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            title=self._title,
        )
        doc.build(self._elements)
        return buf.getvalue()
