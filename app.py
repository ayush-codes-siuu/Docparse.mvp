import json
import os
import sys

import streamlit as st
import pandas as pd
import pdfplumber
from PIL import Image

st.set_page_config(
    page_title="Parserix - GST Invoice Extractor",
    page_icon="\U0001f4c4",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Beta Access Gate
# ---------------------------------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        "<div style='text-align:center; padding-top:80px;'>"
        "<h1>\U0001f4c4 Parserix</h1>"
        "<p style='font-size:1.15rem; color:gray;'>"
        "AI-Powered GST Invoice Extraction — Beta Access Only"
        "</p></div>",
        unsafe_allow_html=True,
    )

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown("")  # spacer
        access_code = st.text_input(
            "Beta Access Code",
            type="password",
            placeholder="Enter your access code",
        )

        if st.button("Unlock Access", use_container_width=True, type="primary"):
            try:
                passcode = st.secrets["BETA_PASSCODE"]
            except (FileNotFoundError, KeyError):
                st.error(
                    "Server misconfiguration: BETA_PASSCODE not set in secrets."
                )
                st.stop()

            if access_code == passcode:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid Access Code. Please contact the team for access.")

        st.caption("Don't have a code? Reach out to the Parserix team.")

    st.stop()


# =========================================================================
#  AUTHENTICATED — Main App below
# =========================================================================

# ---------------------------------------------------------------------------
# Sidebar: About info (API key resolved from secrets/env only)
# ---------------------------------------------------------------------------

# Resolve API key silently: secrets > env var
api_key = os.environ.get("GROQ_API_KEY")
try:
    if not api_key and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    pass

with st.sidebar:
    st.header("About")
    st.markdown(
        "**Parserix** extracts key fields from Indian GST invoices "
        "using AI-powered document understanding."
    )
    st.markdown(
        "**Extracted Fields:**\n"
        "- Vendor Name\n"
        "- Vendor GSTIN\n"
        "- Invoice Number\n"
        "- Invoice Date\n"
        "- Total Taxable Value\n"
        "- Total GST Amount\n"
        "- Grand Total"
    )
    st.divider()
    st.caption("Powered by Groq · Llama 4 Scout 17B")

# ---------------------------------------------------------------------------
# Tesseract availability check (lazy — only when OCR is actually needed)
# ---------------------------------------------------------------------------

def _check_tesseract() -> bool:
    """Return True if Tesseract is available, False otherwise."""
    try:
        import pytesseract
        # Try common Windows install path if not on PATH
        if sys.platform == "win32":
            default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.isfile(default):
                pytesseract.pytesseract.tesseract_cmd = default
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a PDF using pdfplumber. Falls back to OCR for scanned PDFs."""
    text_parts = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    if text_parts:
        return "\n".join(text_parts)

    # Fallback: render pages as images via PyMuPDF and OCR them
    return _ocr_scanned_pdf(uploaded_file)


def _ocr_scanned_pdf(uploaded_file) -> str:
    """Render PDF pages to images with PyMuPDF, then OCR with Tesseract."""
    import fitz  # PyMuPDF
    import pytesseract

    if not _check_tesseract():
        raise EnvironmentError(
            "Tesseract OCR is not installed. This scanned PDF requires OCR.\n\n"
            "**Install Tesseract on Windows:**\n"
            "1. Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
            "2. Install and add to your system PATH.\n"
            "3. Restart this app."
        )

    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text_parts = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_text = pytesseract.image_to_string(img, lang="eng")
        if page_text.strip():
            text_parts.append(page_text)
    doc.close()

    if not text_parts:
        raise ValueError(
            "Could not extract any text from this scanned PDF. "
            "Please ensure the document is legible."
        )
    return "\n".join(text_parts)


def extract_text_from_image(uploaded_file) -> str:
    """Extract text from an image using Pillow + Tesseract OCR."""
    import pytesseract

    if not _check_tesseract():
        raise EnvironmentError(
            "Tesseract OCR is not installed.\n\n"
            "**Install Tesseract on Windows:**\n"
            "1. Download from https://github.com/UB-Mannheim/tesseract/wiki\n"
            "2. Install and add to your system PATH.\n"
            "3. Restart this app."
        )

    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image, lang="eng")
    if not text.strip():
        raise ValueError(
            "OCR could not extract any text from this image. "
            "Ensure the image is clear and contains readable text."
        )
    return text


# ---------------------------------------------------------------------------
# Confidence display helper
# ---------------------------------------------------------------------------

def _render_confidence(confidence: dict) -> None:
    """Render confidence scores with progress bars and color-coded labels."""
    st.subheader("Confidence Scores")
    for field, score in confidence.items():
        label = field.replace("_", " ").title()
        score = max(0, min(100, int(score)))  # clamp 0-100

        if score > 90:
            icon, level = "🟢", "High"
        elif score >= 70:
            icon, level = "🟡", "Medium"
        else:
            icon, level = "🔴", "Low"

        col_label, col_bar = st.columns([1, 2])
        with col_label:
            st.markdown(f"{icon} **{label}** — {score}% *({level})*")
        with col_bar:
            st.progress(score / 100)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("Parserix")
st.subheader("Automated GST Invoice Data Extraction")
st.markdown(
    "Upload one or more Indian GST Invoices (PDF or image) to extract structured data."
)

uploaded_files = st.file_uploader(
    "Upload Invoices",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Upload up to 10 invoices at once. Supported: PDF, PNG, JPG",
)

if uploaded_files:
    # --- Validate API key before processing ---
    if not api_key:
        st.error(
            "Groq API key is required. "
            "Enter it in the sidebar or set the GROQ_API_KEY environment variable."
        )
        st.stop()

    from extraction.extractor import extract_invoice_fields

    all_results: list[dict] = []  # collect for master CSV
    total = len(uploaded_files)

    progress_bar = st.progress(0, text="Starting extraction...")

    for idx, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        progress_bar.progress(
            (idx) / total,
            text=f"Processing {idx + 1}/{total}: {file_name}",
        )

        with st.expander(f"📄 {file_name}", expanded=(total == 1)):
            col1, col2 = st.columns(2)

            # --- Column 1: Raw text extraction ---
            raw_text = None
            with col1:
                st.subheader("Extracted Text")
                with st.spinner("Extracting text..."):
                    try:
                        file_type = uploaded_file.type
                        if file_type == "application/pdf":
                            raw_text = extract_text_from_pdf(uploaded_file)
                        else:
                            raw_text = extract_text_from_image(uploaded_file)
                        st.text_area(
                            "Raw Text",
                            raw_text,
                            height=350,
                            disabled=True,
                            key=f"raw_{idx}",
                        )
                    except (ValueError, EnvironmentError) as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Text extraction failed: {e}")

            # --- Column 2: AI extraction ---
            if raw_text:
                with col2:
                    st.subheader("Structured Output")
                    with st.spinner("Analysing invoice with AI..."):
                        try:
                            result = extract_invoice_fields(raw_text, api_key=api_key)

                            # Separate confidence from fields
                            confidence = result.pop("confidence", {})
                            st.json(result)

                            # Confidence display
                            if confidence:
                                _render_confidence(confidence)

                            # Per-file JSON download
                            export_data = {**result, "confidence": confidence}
                            st.download_button(
                                label="Download JSON",
                                data=json.dumps(export_data, indent=2),
                                file_name=f"invoice_{result.get('invoice_number') or 'unknown'}.json",
                                mime="application/json",
                                key=f"dl_{idx}",
                            )

                            # Add to aggregate (include source filename)
                            flat = {"source_file": file_name, **result}
                            # Flatten confidence into the row
                            for k, v in confidence.items():
                                flat[f"confidence_{k}"] = v
                            all_results.append(flat)

                        except (ValueError, RuntimeError) as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {e}")

    progress_bar.progress(1.0, text=f"Done — {total} invoice(s) processed.")

    # -----------------------------------------------------------------------
    # Master Summary — aggregate all results
    # -----------------------------------------------------------------------
    if all_results:
        st.divider()
        st.header("📊 Master Summary")
        master_df = pd.DataFrame(all_results)

        # Prettify column names for display
        display_df = master_df.copy()
        display_df.columns = [
            col.replace("_", " ").title() for col in display_df.columns
        ]
        st.dataframe(display_df, use_container_width=True)

        # Master CSV download
        csv_data = master_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Master CSV",
            data=csv_data,
            file_name="parserix_master_export.csv",
            mime="text/csv",
            key="master_csv",
        )
