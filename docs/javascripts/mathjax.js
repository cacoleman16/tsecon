/* MathJax 3 configuration for mkdocs-material + pymdownx.arithmatex.
 *
 * Must load BEFORE the MathJax bundle (see extra_javascript order in
 * mkdocs.yml). Only nodes arithmatex marked at build time are processed.
 * The document$ subscription exists for page swaps (search navigation,
 * instant loading if ever enabled): it typesets ONLY arithmatex nodes that
 * are not already rendered, so it can never double-render the equations the
 * bundle's automatic initial typeset already handled.
 */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  /* Pre-bundle emissions see the plain config object — skip them. */
  if (!window.MathJax.typesetPromise || !window.MathJax.startup) return;
  MathJax.startup.promise.then(() => {
    const pending = Array.from(
      document.querySelectorAll(".arithmatex")
    ).filter((node) => !node.querySelector("mjx-container"));
    if (pending.length) MathJax.typesetPromise(pending);
  });
});
