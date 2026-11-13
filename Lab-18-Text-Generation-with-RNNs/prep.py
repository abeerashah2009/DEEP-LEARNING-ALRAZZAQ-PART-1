text = open('alice_in_wonderland.txt', encoding='utf-8').read()
text = text.lower()
text = text.replace('\r', '').replace('\n', ' ')
text = text[:30000]
open('corpus.txt', 'w', encoding='utf-8').write(text)
print(f'Text length: {len(text)} characters')
