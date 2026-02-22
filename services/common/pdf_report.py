"""
services/common/pdf_report.py
Generates Institutional Tear Sheets for investors.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

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
            with open(output_path.replace('.pdf', '.md'), 'w') as f:
                f.write("# APEX TRADING SYSTEM - TEAR SHEET\n\n")
                for k, v in metrics.items():
                    f.write(f"* **{k}**: {v}\n")
            return False

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Title
        elements.append(Paragraph("APEX ALGORITHMIC TRADING - INSTITUTIONAL TEAR SHEET", styles['Title']))
        elements.append(Spacer(1, 20))
        
        # Subtitle
        elements.append(Paragraph("VERIFIED TRACK RECORD & RISK METRICS", styles['Heading2']))
        elements.append(Spacer(1, 10))

        # Metrics Table
        data = [["Metric", "Apex System Value", "Industry Benchmark"]]
        
        # Standardize Pitch Metrics
        for key, val in metrics.items():
            data.append([key, str(val), "N/A"])

        t = Table(data, colWidths=[200, 150, 150])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('FONTSIZE', (0,1), (-1,-1), 10),
        ]))
        
        elements.append(t)
        
        # Footer
        elements.append(Spacer(1, 40))
        elements.append(Paragraph("<i>Note: Report generated securely via Apex Deterministic Replay Engine.</i>", styles['Normal']))
        
        doc.build(elements)
        logger.info(f"📄 PDF Tear Sheet successfully generated at: {output_path}")
        return True
