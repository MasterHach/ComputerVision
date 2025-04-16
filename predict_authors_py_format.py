# -*- coding: utf-8 -*-
"""
Версия кода в формате .py
"""

# ========== 1 Импорт необходимых библиотек =================
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 2 Подготовка параметров и других переменных =================

chunks = []
authors = []
device = torch.device('cuda')
model_name = "cointegrated/rubert-tiny"  # Легкая версия ruBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_tokens = 512
chunk_size = 5 # количество предложений, которые будут входить в 1 запись в датафрейме для дальнейшего обучения
author_dir = 'authors' # название папки с обучающими данными
train_dir = 'train' # название папки с тестовыми данными
stop_words = set(stopwords.words('russian'))

# ========== 3 Сбор и преобработка данных =================

# проходим по всем 6 файлам, преобрабатываем текст, разделяем на предложения, добавляем в датафрейм
for file in os.listdir(author_dir):
    """
    Не было проведено аугментации, удаления редких слов, удаления символов, лемминга и стемминга, так как
    показалось, что удалив может пропасть авторский стиль (например много тире в тексте - значит диалог, авторский прием,
    много ! в тексте - тоже не у каждого автора)
    """
    author = file.split('.')[0]
    path = os.path.join(author_dir, file)
    with open(path, encoding='utf-8') as f:
        text = f.read()
        text = text.lower() # преобразование всего текста в нижний регистр чтобы не было повторений например Текст, текст
        text = " ".join([word for word in text.split() if word not in stop_words]) # удаление разных и в на к от (неинформативно)
        sentences = sent_tokenize(text, language="russian")
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i+chunk_size])
            if len(tokenizer(chunk)["input_ids"]) <= max_tokens:
                chunks.append(chunk)
                authors.append(author)

df = pd.DataFrame({
    'chunk': chunks,
    'author': authors
})

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6).to("cuda")

# Функция для токенизации текста
def tokenize_function(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=512)

# Применение токенизации ко всем строкам
df['input_ids'] = df['chunk'].apply(lambda x: tokenize_function(x)['input_ids'])
df['attention_mask'] = df['chunk'].apply(lambda x: tokenize_function(x)['attention_mask'])

author_to_id = {'Bradbury': 0, 'Bulgakov': 1, 'Fry': 2, 'Genri': 3, 'Simak': 4, 'Strugatskie': 5}
df['labels'] = df['author'].map(author_to_id)

# Разделяем на обучающую и тестовую выборки
train_texts, val_texts, train_labels, val_labels = train_test_split(df['chunk'], df['labels'], test_size=0.2, shuffle=True)

# Класс для подготовки разбитых на чанки данных (в т.ч токенизация) для обучения модели
class AuthorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Создаём датасеты для обучения
train_dataset = AuthorDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = AuthorDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)


# ================ 4 Обучение модели ================


# Параметры обучения
training_args = TrainingArguments(
    output_dir='./results',          # Куда сохранять модель
    num_train_epochs=3,              # Количество эпох
    per_device_train_batch_size=8,   # Размер батча для тренировки
    per_device_eval_batch_size=8,    # Размер батча для валидации
    warmup_steps=500,                # Количество шагов разогрева
    weight_decay=0.01,               # Регуляризация
    logging_dir='./logs',            # Логирование
    logging_steps=10,
    eval_strategy="epoch",           # Оценка после каждой эпохи
    fp16=True
)

# Создаем Trainer
trainer = Trainer(
    model=model,                         # Модель
    args=training_args,                  # Параметры обучения
    train_dataset=train_dataset,         # Обучающий датасет
    eval_dataset=val_dataset             # Валидационный датасет
)

# Обучаем модель
trainer.train()


# ================ 5 Проверка работоспособности модели на тестовых данных ====================

predictions = []

# Сбор данных из папки с тестовыми данными и сразу же совершение предсказания

for file in os.listdir(train_dir):
  with open(os.path.join(train_dir, file), "r", encoding="utf-8") as f:
        text = f.read()

  inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
  with torch.no_grad():
      outputs = model(**inputs)
      probs = torch.nn.functional.softmax(outputs.logits, dim=1)
      predicted_class = torch.argmax(probs, dim=1).item()

  predicted_author = [k for k, v in author_to_id.items() if v == predicted_class]
  predictions.append((file, "".join(predicted_author).lower()))

# Сохранение данных в датафрейм и выгрузка в csv

df = pd.DataFrame(predictions, columns=["filename", "author"])
df.to_csv("predictions.csv", index=False)

# ================ 6 Проверка точности модели ======================

pred_df = pd.read_csv("predictions.csv")
true_df = pd.read_csv("corrects.csv") # файл где содержатся верные авторы отрывков 

# Объединим по имени файла
merged = pd.merge(pred_df, true_df, on='filename')
merged.columns = ['filename', 'predicted_author', 'true_author', 'class']

y_true = merged['true_author']
y_pred = merged['predicted_author']

# Вывод метрик для оценки

print("🔎 Accuracy:", accuracy_score(y_true, y_pred))
print("🔎 F1-score (macro):", f1_score(y_true, y_pred, average='macro'))

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

# Отображение матрицы ошибок

labels = sorted(y_true.unique())

cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Матрица ошибок")
plt.tight_layout()
plt.show()

# График F1-score по каждому классу (автору)

report = classification_report(y_true, y_pred, output_dict=True)
f1_scores = {label: report[label]['f1-score'] for label in labels}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), palette='viridis')
plt.ylabel('F1-score')
plt.title('F1-score по авторам')
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()