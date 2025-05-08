import os

# Путь к папке с UECFoodPix
BASE = os.path.join('../testML', 'data', 'UECFoodPIXCOMPLETE')
CAT_FILE = os.path.join('../testML', 'data', 'category.txt')
OUT_YAML = 'data.yaml'

# Проверим, что всё на месте
assert os.path.isdir(BASE), f"Папка не найдена: {BASE}"
assert os.path.isfile(CAT_FILE), f"Файл не найден: {CAT_FILE}"

# Считываем классы
with open(CAT_FILE, encoding='utf-8') as f:
    classes = [l.strip() for l in f if l.strip()]

# Пишем data.yaml
with open(OUT_YAML, 'w', encoding='utf-8') as f:
    f.write(f"path: {BASE}\n")
    f.write("train: train/img\n")
    f.write("val:   test/img\n\n")
    f.write("names:\n")
    for i, name in enumerate(classes):
        # Всегда ровно два пробела перед ключом, ключ без выравнивания по цифрам
        f.write(f"  {i}: \"{name}\"\n")

print(f"Сгенерирован {OUT_YAML} с {len(classes)} классами.")