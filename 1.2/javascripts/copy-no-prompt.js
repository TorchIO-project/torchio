// Strip shell and Python prompts from copied code, similar to sphinx-copybutton.
// When prompt lines are detected, only prompt lines are kept (outputs excluded).
(function () {
  var promptRe = /^(>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: )/;

  function stripPrompts(text) {
    var lines = text.split("\n");
    var hasPrompt = lines.some(function (line) {
      return promptRe.test(line);
    });
    if (!hasPrompt) return text;
    // Keep only prompt lines, stripping the prompt prefix
    return lines
      .filter(function (line) {
        return promptRe.test(line);
      })
      .map(function (line) {
        return line.replace(promptRe, "");
      })
      .join("\n");
  }

  document.addEventListener("copy", function (e) {
    var sel = document.getSelection();
    if (!sel) return;
    var text = sel.toString();
    if (!text) return;
    var cleaned = stripPrompts(text);
    if (cleaned !== text && e.clipboardData) {
      e.clipboardData.setData("text/plain", cleaned);
      e.preventDefault();
    }
  });
})();
