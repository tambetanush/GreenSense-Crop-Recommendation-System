import sys

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    target = """          .then((data) => {
            if (data.error) {
              toast(data.error, "err");
              return;
            }
            
            outBox.innerHTML = buildAIHTML(data);
            outBox.classList.add("open");
            outBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
          })"""

    replacement = """          .then((data) => {
            if (data.error) {
              toast(data.error, "err");
              return;
            }
            
            const bannerHtml = `
             <div style="background: var(--accent-pale); border: 1px solid var(--accent-dim); padding: 8px 12px; border-radius: var(--radius); margin-bottom: 16px; font-size: 13px; display: flex; justify-content: space-between; align-items: center;">
               <span>⚡ <strong>Explanation Ready</strong> (Saved to cache).</span>
               <a href="#" onclick="showForceRefreshModal(event)" style="color: var(--accent); text-decoration: underline; cursor: pointer;">Force New Request</a>
             </div>
            `;
            outBox.innerHTML = bannerHtml + buildAIHTML(data);
            outBox.classList.add("open");
            btn.style.display = "none";
            outBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
          })"""

    if target in content:
        content = content.replace(target, replacement)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'{filepath} patched successfully')
    else:
        print(f'Target not found in {filepath}!')

patch_file(r'e:\Projects\GreenSense_2\templates\recommendation.html')
