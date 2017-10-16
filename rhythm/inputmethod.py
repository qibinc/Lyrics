import json
import pypinyin

print('Loading dictionary...')

characterFrequency = json.load(open('singleCharacterFrequency.json', 'r'))
tupleCharacterFrequency = json.load(open('tupleCharacterFrequency.json', 'r'))
characterFrequency[''] = 0
for key in characterFrequency:
	characterFrequency[''] += characterFrequency[key]
characterFrequency.update(tupleCharacterFrequency)

print('Loading complete!\n')

while True:
	line = raw_input("Please input pinyin:\n")
	pinyinList = line.lower().split()

	for pinyin in pinyinList:
		if not pinyin in pinyinDictionary:
			print("Pinyin error!")
			break
	else:
		result = ProcessSentence(pinyinList)

		if result == '':
			print("Can't find path.")
		else:
			print(result)
