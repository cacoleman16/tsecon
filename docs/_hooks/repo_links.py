"""mkdocs hook: rewrite relative links that point OUTSIDE the docs tree to
absolute GitHub URLs.

The docs are also read directly on GitHub, where relative links like
`../fixtures/var.json` or `../../ROADMAP.md` resolve against the repo. In the
built site those targets live outside `docs/`, so mkdocs can't resolve them.
Rather than maintain two link styles, we keep the repo-relative form in the
Markdown and rewrite the handful that escape `docs/` to `blob/main` URLs at
build time. In-site links (to guide/reference/examples pages) are untouched.
"""
import re

BLOB = "https://github.com/cacoleman16/tsecon/blob/main/"

# ](  one-or-more ../  (fixtures/… | bindings/… | prototypes/… | crates/… | ROADMAP.md)  )
_PATTERN = re.compile(
    r"\]\((?:\.\./)+"
    r"((?:fixtures|bindings|prototypes|crates)/[^)\s]*|ROADMAP\.md)"
    r"\)"
)

# The roadmap module specs live at docs/roadmap/ but are excluded from the
# built site (they are internal planning docs); links to them go to GitHub.
_ROADMAP = re.compile(r"\]\((?:\.\./)*roadmap/([^)\s]*)\)")


def on_page_markdown(markdown, page=None, config=None, files=None):
    markdown = _PATTERN.sub(lambda m: f"]({BLOB}{m.group(1)})", markdown)
    markdown = _ROADMAP.sub(lambda m: f"]({BLOB}docs/roadmap/{m.group(1)})", markdown)
    return markdown
