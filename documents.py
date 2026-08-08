"""Safe PDF ingestion with page-level provenance for the RAG index."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import unquote, urlparse

import PyPDF2
import requests


DEFAULT_DOCUMENTS_DIR = Path("./documents")
MAX_PDF_BYTES = 25 * 1024 * 1024


def safe_pdf_filename(name: str) -> str:
    """Return a basename-only PDF filename suitable for local storage."""
    cleaned = Path(unquote(name)).name.replace("\x00", "").strip()
    if not cleaned.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    if cleaned in {"", ".pdf"}:
        raise ValueError("Invalid PDF filename.")
    return cleaned


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_pdf_files(folder_path: str = "./documents") -> List[str]:
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    return [str(path) for path in sorted(folder.iterdir()) if path.is_file() and path.suffix.lower() == ".pdf"]


def extract_pages_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text page by page so every retrieved chunk can cite a page."""
    pages: List[Dict[str, Any]] = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page_number": page_number, "text": text})
    return pages


def extract_text_from_pdf(pdf_path: str) -> str:
    return "\n\n".join(page["text"] for page in extract_pages_from_pdf(pdf_path))


def _build_document(file_path: str, doc_id: str | None = None, source_url: str = "") -> Dict[str, Any] | None:
    pages = extract_pages_from_pdf(file_path)
    if not pages:
        return None
    filename = Path(file_path).name
    digest = _file_sha256(file_path)
    return {
        "document_id": doc_id or digest[:16],
        "title": Path(filename).stem.replace("_", " ").replace("-", " "),
        "doc_type": "reference_pdf",
        "effective_date": "",
        "current_status": "unknown",
        "domain": "Nuclear Law",
        "department": "User Uploads",
        "file_path": file_path,
        "source_url": source_url,
        "sha256": digest,
        "pages": pages,
        "text": "\n\n".join(page["text"] for page in pages),
    }


def get_documents() -> List[Dict[str, Any]]:
    docs = []
    for pdf_path in get_pdf_files():
        try:
            doc = _build_document(pdf_path)
        except Exception as exc:
            print(f"Error reading {pdf_path}: {exc}")
            continue
        if doc:
            docs.append(doc)
    return docs


def _validate_public_https_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("Only public HTTPS PDF URLs are allowed.")
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URLs are not allowed.")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror as exc:
        raise ValueError("The URL hostname could not be resolved.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ValueError("Private or local network URLs are not allowed.")


def download_pdf_from_url(url: str, folder: str = "./documents") -> str:
    """Download a public HTTPS PDF with redirect and size protections."""
    _validate_public_https_url(url)
    parsed = urlparse(url)
    filename = safe_pdf_filename(Path(parsed.path).name)
    destination_dir = Path(folder)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    temporary = destination.with_suffix(destination.suffix + ".part")

    response = requests.get(url, stream=True, timeout=(5, 30), allow_redirects=False)
    response.raise_for_status()
    if 300 <= response.status_code < 400:
        raise ValueError("Redirecting PDF URLs are rejected; provide the final HTTPS URL.")
    content_type = response.headers.get("Content-Type", "").lower()
    if "pdf" not in content_type and not parsed.path.lower().endswith(".pdf"):
        raise ValueError("The URL did not return a PDF.")
    declared_size = int(response.headers.get("Content-Length", "0") or 0)
    if declared_size > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 25 MiB limit.")

    written = 0
    try:
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                written += len(chunk)
                if written > MAX_PDF_BYTES:
                    raise ValueError("PDF exceeds the 25 MiB limit.")
                handle.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination)


def load_pdf(file_path: str, doc_id: str | None = None, source_url: str = "") -> Dict[str, Any] | None:
    return _build_document(file_path, doc_id=doc_id, source_url=source_url)
