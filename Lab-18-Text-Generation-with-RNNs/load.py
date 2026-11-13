raw = open('alice_raw.txt', encoding='utf-8-sig').read()
start = raw.index('*** START OF')
start = raw.index('\n', start) + 1
end = raw.index('*** END OF')
text = raw[start:end].strip()
open('alice_in_wonderland.txt', 'w', encoding='utf-8').write(text)

print(f'Text length: {len(text)} characters')
