def classify_document(text: str):
    t = text.lower()

    # -------- QUIZ OR KNOWLEDGE --------
    if any(k in t for k in ["quiz", "mcq", "multiple choice", "questions", "answer key"]):
        doc_type = "RESTRICTED_QUIZ"
    else:
        doc_type = "KNOWLEDGE"

    # -------- ROLE --------
    if any(k in t for k in [
        "general policy", "company policy", "it access", "system access",
        "code of conduct", "security policy"
    ]):
        role = "UNIVERSAL"
    elif any(k in t for k in ["hr", "leave policy", "payroll", "attendance"]):
        role = "HR"
    elif any(k in t for k in ["developer", "api", "backend", "frontend", "code"]):
        role = "DEVELOPER"
    elif "intern" in t:
        role = "INTERN"
    else:
        role = "UNIVERSAL"

    return doc_type, role
