# Sentiment Analysis on IMDB Dataset

## Project Overview
This project demonstrates a sentiment analysis model built using a deep learning approach. The model classifies movie reviews from the IMDB dataset as positive or negative. The reviews are preprocessed, tokenized, and fed into a Bi-directional LSTM (Long Short-Term Memory) neural network for classification.

---

## Dataset
The dataset used in this project is the **IMDB Dataset of 50K Movie Reviews**. Each review is labeled as positive or negative.

### Key Information:
- **Columns:**
  - `review`: Contains the text of the review.
  - `sentiment`: Indicates the sentiment of the review ("positive" or "negative").

---

## Project Workflow

### 1. **Data Loading**
The dataset is loaded using pandas:
```python
df = pd.read_csv(r'D:\Documents\Nexmedis Salsa\IMDB Dataset.csv')
```

### 2. **Data Cleaning**
Text data is cleaned to remove HTML tags, non-alphabetic characters, and converted to lowercase. A custom function, `clean_text`, is used:
```python
def clean_text(text):
    text = re.sub(r'<br />', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

df['cleaned_review'] = df['review'].apply(clean_text)
```

### 3. **Label Encoding**
The `sentiment` column is encoded into numeric values:
- **0**: Negative
- **1**: Positive
```python
label_encoder = LabelEncoder()
df['sentiment_numeric'] = label_encoder.fit_transform(df['sentiment'])
```

### 4. **Train-Test Split**
The dataset is split into training and testing sets (80%-20% split):
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5. **Text Tokenization and Padding**
Using Keras’ Tokenizer, the reviews are converted into sequences and padded to ensure uniform input size:
```python
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_seq, maxlen=200, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=200, padding='post', truncating='post')
```

### 6. **Model Architecture**
A Bi-directional LSTM is used with regularization and dropout to prevent overfitting. The architecture includes:
- **Embedding Layer:** Converts words into dense vectors of fixed size.
- **Bi-directional LSTM Layers:** Captures context from both directions in the text.
- **Dense Layers:** Processes the LSTM outputs.
- **Sigmoid Output Layer:** Produces binary classification output.
```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=100, input_length=200),

    Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))),
    Dropout(0.5),

    Bidirectional(LSTM(64, kernel_regularizer=l2(0.01))),
    Dropout(0.5),

    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### 7. **Model Training**
The model is trained using early stopping to avoid overfitting:
```python
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

history = model.fit(
    X_train_padded,
    y_train,
    epochs=10,
    validation_data=(X_test_padded, y_test),
    batch_size=32,
    callbacks=[early_stop]
)
```

### 8. **Model Evaluation**
The model's performance is evaluated on the test set:
```python
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
```

---

## Results
- **Test Accuracy:** Achieved an accuracy of around 85%-90% (depending on the dataset split and training).
- **Test Loss:** Monitored to ensure no overfitting.

### Training and Validation Metrics
The training and validation accuracy/loss curves are visualized:
```python
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')

plt.show()
```
