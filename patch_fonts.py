import sys

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # We want to add font-family: var(--serif); to the div and the a tag
    # Let's replace the common style string block.
    
    # Block 1 (In renderExplain)
    target1 = 'margin-bottom: 16px; font-size: 13px; display: flex;'
    replacement1 = 'margin-bottom: 16px; font-size: 15px; font-family: var(--serif); display: flex;'
    
    content = content.replace(target1, replacement1)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'{filepath} patched successfully')

patch_file(r'e:\Projects\GreenSense_2\templates\index.html')
patch_file(r'e:\Projects\GreenSense_2\templates\recommendation.html')
