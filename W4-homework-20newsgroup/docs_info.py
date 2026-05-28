import os

# 数据目录
data_dir = r".\data"

# 定义类别（对应文件名）
categories = [
    "comp.graphics",
    "comp.os.ms-windows.misc", 
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "misc.forsale",
    "talk.politics.misc",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.religion.misc",
    "alt.atheism",
    "soc.religion.christian"
]

# 加载数据 - 每个文件包含多个文档，用 "From:" 分隔
documents = []
labels = []
label_names = []

for idx, cat in enumerate(categories):
    filepath = os.path.join(data_dir, f"{cat}.txt")
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            docs = content.split("\nFrom: ")
            for doc in docs:
                doc = doc.strip()
                if doc:
                    if not doc.startswith("From:"):
                        doc = "From: " + doc
                    documents.append(doc)
                    labels.append(idx)
        label_names.append(cat)

print(f"总文档数：{len(documents)}")
print(f"类别数：{len(label_names)}")
print(f"类别列表：{label_names}")
