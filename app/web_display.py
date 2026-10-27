"""Display-only protection for evidence identifiers and untrusted filenames."""
from html import escape
import re


def citation_markdown(text: str) -> str:
    # Keep Markdown rendering; render only identifiers as inline code.
    return re.sub(r"(?<!`)\[S[1-9]\d*\](?!`)", lambda match: f"`{match[0]}`", text)


def evidence_html(source: dict) -> str:
    # No model-generated HTML. Escape every dynamic field before using st.html.
    text = (f"{source['evidence_label']}  {source['filename']} · "
            f"第 {source['page_number']} 页 · {source['chunk_id']}")
    return f'<div translate="no" class="notranslate"><code>{escape(text)}</code></div>'
