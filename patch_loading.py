import sys

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacement 1: The loading state
    target1 = """          btn.disabled = true;
          btn.innerHTML = `<span class="spinner" style="border-color:currentColor;border-top-color:transparent;width:12px;height:12px"></span> Generating...`;
          
          // Hide previous output
          outBox.classList.remove("open");
          outBox.innerHTML = "";"""

    replacement1 = """          if (btn) btn.style.display = "none";
          
          outBox.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 48px 0; color: var(--text-muted); gap: 16px;">
              <span class="spinner" style="width: 32px; height: 32px; border-width: 3px; border-color: var(--accent); border-top-color: transparent;"></span>
              <div style="font-family: var(--serif); font-size: 16px; color: var(--accent);">Generating AI Explanation...</div>
            </div>
          `;
          outBox.classList.add("open");"""

    # Replacement 2: The catch/finally block
    target2 = """            .catch(() => toast("AI Explain request failed", "err"))
            .finally(() => {
              btn.innerHTML = `✨ AI Explain`;
              btn.disabled = false;
            });"""

    replacement2 = """            .catch(() => {
              toast("AI Explain request failed", "err");
              outBox.innerHTML = "";
              outBox.classList.remove("open");
              if (btn) {
                  btn.innerHTML = `✨ AI Explain`;
                  btn.disabled = false;
                  btn.style.display = "inline-flex";
              }
            });"""

    changed = False
    if target1 in content:
        content = content.replace(target1, replacement1)
        changed = True
    else:
        print(f"Target 1 not found in {filepath}")

    if target2 in content:
        content = content.replace(target2, replacement2)
        changed = True
    else:
        print(f"Target 2 not found in {filepath}")

    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'{filepath} patched successfully')

patch_file(r'e:\Projects\GreenSense_2\templates\index.html')
patch_file(r'e:\Projects\GreenSense_2\templates\recommendation.html')
