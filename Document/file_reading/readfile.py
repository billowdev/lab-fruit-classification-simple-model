from docx import Document

document = Document('prediction-result.docx')

print(document.paragraps)


# with open('prediction-result.docx') as f:
# 	lines = f.read()

# 	print(lines)