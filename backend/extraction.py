import pymupdf
import re

def extract_pdf_text(file_bytes: bytes) -> dict:
    """
    Extract text from every page of a PDF.

    Page-level text is preserved because TenderIQ will later
    use page references for RAG citations and evidence.
    """

    try:
        document = pymupdf.open(
            stream=file_bytes,
            filetype="pdf",
        )
    except Exception as exc:
        raise ValueError(
            "The uploaded file is not a valid PDF."
        ) from exc

    pages = []

    try:
        for page_number, page in enumerate(document, start=1):

            raw_text = page.get_text(
                "text",
                sort=True,
            ).strip()

            cleaned_text = clean_text(raw_text)

            pages.append(
                {
                    "page": page_number,
                    "text": cleaned_text,
                }
            )

    finally:
        document.close()

    full_text = "\n\n".join(
        page["text"]
        for page in pages
        if page["text"]
    )

    return {
        "page_count": len(pages),
        "text": full_text,
        "pages": pages,
    }


def clean_text(text: str) -> str:
    """
    Clean and normalize text extracted from a PDF.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")

    # Remove excessive spaces while preserving newlines.
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove spaces around newlines.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def find_value(text: str, pattern: str) -> str | None:
    """
    Extract a single value from a labeled field.

    Example:
    'Tender Reference: TND-2026-014'
    """

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if not match:
        return None

    value = match.group(1).strip()

    return value if value else None


def extract_metadata(pages: list[dict]) -> dict:
    """
    Extract deterministic metadata from the tender.

    Handles common variations in tender terminology while
    keeping extraction rule-based and predictable.
    """

    full_text = "\n".join(
        page["text"]
        for page in pages
        if page["text"]
    )

    # Project name
    project_name = find_value(
        full_text,
        r"Project\s+Name\s*:\s*(.+)",
    )

    # Some tenders use the document title instead of
    # an explicit "Project Name:" field.
    if not project_name:
        title_match = re.search(
            r"Invitation\s+to\s+Tender\s*\n+(.+?)(?:\n|$)",
            full_text,
            flags=re.IGNORECASE,
        )

        if title_match:
            project_name = title_match.group(1).strip()

    # Some documents use "REQUEST FOR PROPOSAL"
    # followed immediately by the project title.
    if not project_name:
        title_match = re.search(
            r"Request\s+for\s+Proposal\s*\n+(.+?)(?:\n|$)",
            full_text,
            flags=re.IGNORECASE,
        )

        if title_match:
            project_name = title_match.group(1).strip()

    # Tender reference
    tender_reference = find_value(
        full_text,
        r"(?:Tender\s+Reference|Tender\s+No\.?|Tender\s+Number|RFP\s+No\.?)\s*:\s*(.+)",
    )

    # Issuing organization / client
    issuing_organization = find_value(
        full_text,
        r"(?:Issuing\s+Organization|Client|Employer|Procuring\s+Entity)\s*:\s*(.+)",
    )

    # Project location
    location = find_value(
        full_text,
        r"(?:Project\s+Location|Project\s+Site|Location|Site)\s*:\s*(.+)",
    )

    # Submission deadline
    submission_deadline = find_value(
        full_text,
        r"(?:Submission\s+Deadline|Bid\s+Submission\s+Deadline|"
        r"Tender\s+Submission\s+Deadline)\s*:\s*(.+)",
    )

    # Contract value / project budget
    contract_value = find_value(
        full_text,
        r"(?:Estimated\s+Contract\s+Value|Estimated\s+Project\s+Budget|"
        r"Contract\s+Value|Project\s+Value|Budget)\s*:\s*(.+)",
    )

    return {
        "project_name": project_name,
        "tender_reference": tender_reference,
        "issuing_organization": issuing_organization,
        "location": location,
        "submission_deadline": submission_deadline,
        "contract_value": contract_value,
    }


def extract_requirements(pages: list[dict]) -> list[dict]:
    """
    Extract mandatory tender requirements while preserving
    the page where each requirement was found.

    Supports common tender formats where requirements appear
    in a section containing a Requirement / Mandatory table.
    """

    requirements = []

    for page in pages:
        page_text = page["text"]

        # Look for the mandatory requirements section.
        section_match = re.search(
            r"(?:Mandatory\s+Requirements|Eligibility\s+and\s+Mandatory\s+Requirements)"
            r"(.*?)(?=\n\s*\d+\.\s+[A-Z][^\n]*|\Z)",
            page_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not section_match:
            continue

        section_text = section_match.group(1)

        # Match requirement rows ending with "Yes".
        #
        # Example:
        # CAC registration and valid company documentation Yes
        #
        # The non-greedy match prevents multiple rows from
        # accidentally being combined.
        matches = re.finditer(
            r"([A-Za-z][^\n]+?)\s+(Yes|No)\s*$",
            section_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        for match in matches:
            requirement = match.group(1).strip()
            mandatory_value = match.group(2).strip().lower()

            # Clean excessive whitespace.
            requirement = re.sub(
                r"\s+",
                " ",
                requirement,
            )

            # Ignore table headers.
            if requirement.lower() in {
                "requirement",
                "mandatory",
            }:
                continue

            item = {
                "requirement": requirement,
                "mandatory": mandatory_value == "yes",
                "page": page["page"],
            }

            # Prevent duplicates.
            already_exists = any(
                existing["requirement"].lower() == requirement.lower()
                for existing in requirements
            )

            if not already_exists:
                requirements.append(item)

    return requirements


def extract_submission_documents(pages: list[dict]) -> list[dict]:
    """Extract required tender submission documents with page references."""

    documents = []

    for page in pages:
        page_text = page["text"]

        section_match = re.search(
            r"(?:Required Submission Documents|Submission Documents)"
            r"(.*?)(?=\n\s*\d+\.\s+[A-Z]|\Z)",
            page_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not section_match:
            continue

        section_text = section_match.group(1)

        # Match numbered submission items.
        matches = re.finditer(
            r"(?:^|\n)\s*\d+\.\s*(.+?)(?=\n\s*\d+\.|\Z)",
            section_text,
            flags=re.DOTALL,
        )

        for match in matches:
            document = re.sub(r"\s+", " ", match.group(1)).strip()

            if document:
                documents.append({
                    "document": document,
                    "page": page["page"],
                })

    return documents


def extract_commercial(pages: list[dict]) -> dict:
    """Extract commercial and contractual conditions."""

    full_text = "\n".join(
        page["text"]
        for page in pages
        if page["text"]
    )

    return {
        "construction_period": find_value(
            full_text,
            r"(?:Construction\s+Period|Contract\s+Duration)\s*:\s*(.+)",
        ),

        "payment_terms": find_value(
            full_text,
            r"(?:Payment\s+Terms|Payment\s+Terms\s+and\s+Conditions)\s*:\s*(.+)",
        ),

        "performance_bond": find_value(
            full_text,
            r"(?:Performance\s+Bond)\s*:\s*(.+)",
        ),

        "advance_payment_guarantee": find_value(
            full_text,
            r"(?:Advance\s+Payment\s+Guarantee)\s*:\s*(.+)",
        ),
    }
    
    
def extract_evaluation(pages: list[dict]) -> dict:
    """Extract tender evaluation criteria."""

    full_text = "\n".join(
        page["text"]
        for page in pages
        if page["text"]
    )

    evaluation = {}

    criteria = {
        "technical_capability": r"Technical\s+Capability\s*:\s*(.+)",
        "experience": r"Experience\s*:\s*(.+)",
        "methodology": r"Methodology\s*:\s*(.+)",
        "programme": r"Programme\s*:\s*(.+)",
        "safety": r"(?:Safety|Health\s+and\s+Safety)\s*:\s*(.+)",
        "quality": r"Quality\s*:\s*(.+)",
        "financial_capacity": r"Financial\s+Capacity\s*:\s*(.+)",
        "commercial_value": r"Commercial\s+Value\s*:\s*(.+)",
    }

    for key, pattern in criteria.items():
        evaluation[key] = find_value(full_text, pattern)

    return evaluation


def extract_risks(pages: list[dict]) -> list[dict]:
    """
    Extract explicitly mentioned project risks and constraints
    while preserving their source page.

    Supports common tender sections such as:
    - Key Risks and Constraints
    - Risks and Constraints
    - Key Risks
    """

    risks = []

    for page in pages:
        page_text = page["text"]

        # Find the risks/constraints section.
        section_match = re.search(
            r"(?:Key\s+Risks\s+and\s+Constraints|"
            r"Risks\s+and\s+Constraints|"
            r"Key\s+Risks)"
            r"(.*?)(?=\n\s*\d+\.\s+[A-Z][^\n]*|\Z)",
            page_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if not section_match:
            continue

        section_text = section_match.group(1)

        # Remove introductory language that is not itself a risk.
        section_text = re.sub(
            r"^(?:\s*Contractors?\s+should\s+account\s+for|"
            r"\s*Bidders?\s+should\s+consider)\s+",
            "",
            section_text,
            flags=re.IGNORECASE,
        )

        # Normalize whitespace.
        section_text = re.sub(
            r"\s+",
            " ",
            section_text,
        ).strip()

        # Common risk phrases.
        #
        # We intentionally keep this controlled rather than asking
        # an LLM to invent or infer risks during Day 1.
        risk_patterns = [
            r"material price fluctuations",
            r"inflation and fluctuations in construction material prices",
            r"weather-related delays",
            r"site access restrictions",
            r"restricted access to occupied clinical areas",
            r"procurement lead times",
            r"long lead times for imported medical equipment",
            r"coordination between multiple subcontractors",
            r"utility interruptions",
            r"coordination with hospital operations",
        ]

        for pattern in risk_patterns:
            matches = re.finditer(
                pattern,
                section_text,
                flags=re.IGNORECASE,
            )

            for match in matches:
                risk = match.group(0).strip()

                risk = re.sub(
                    r"\s+",
                    " ",
                    risk,
                )

                already_exists = any(
                    item["risk"].lower() == risk.lower()
                    for item in risks
                )

                if not already_exists:
                    risks.append(
                        {
                            "risk": risk,
                            "page": page["page"],
                        }
                    )

    return risks


def extract_tender_intelligence(pages: list[dict]) -> dict:
    """
    Combine deterministic extraction into structured
    TenderIQ intelligence.
    """

    metadata = extract_metadata(pages)
    requirements = extract_requirements(pages)
    submission_documents = extract_submission_documents(pages)
    risks = extract_risks(pages)
    commercial = extract_commercial(pages)
    evaluation = extract_evaluation(pages)

    mandatory_requirements = [
        item
        for item in requirements
        if item["mandatory"]
    ]

    return {
        "metadata": metadata,

        "requirements": {
            "mandatory": mandatory_requirements,
            "all": requirements,
            "submission_documents": submission_documents,
        },

        "risks": risks,

        "commercial": commercial,

        "evaluation": evaluation,
    }