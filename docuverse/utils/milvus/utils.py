import re
def sanitize_milvus_collection_name(name: str, replacement: str = "_") -> str:
    """
    Sanitize a string to be a valid Milvus collection name.

    Milvus collection names can only contain letters, numbers, and underscores.

    Args:
        name: The original collection name
        replacement: Character to replace invalid characters with (default: "_")

    Returns:
        A sanitized collection name
    """
    # Replace any character that's not a letter, number, or underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', replacement, name)

    # Remove consecutive replacement characters
    sanitized = re.sub(f'{re.escape(replacement)}+', replacement, sanitized)

    # Remove leading/trailing replacement characters
    sanitized = sanitized.strip(replacement)

    # Ensure the name is not empty
    if not sanitized:
        sanitized = "collection_" + str(hash(name))[:8]

    return sanitized
