import json
import codecs
import re
  
with open('./data/kpi-java.json', encoding='utf-8') as fh:
    data = json.load(fh)


response = json.dumps(data, ensure_ascii=False).replace("},", "},\n").replace("{", "\n{", 1)

# replace last
def rreplace(s, old, new, count):
    return (s[::-1].replace(old[::-1], new[::-1], count))[::-1]

response = rreplace(response, "}", "}\n", 1)


# save
writefile = codecs.open('./data/processed-messages-kpi-java.json', 'w', 'utf-8')

writefile.write(response)

