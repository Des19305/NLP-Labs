from pathlib import Path
import csv
import numpy as np


def softmax_stable(x, axis=-1):
    """Softmax ổn định số bằng cách trừ max theo trục chuẩn hóa."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("Không thể tính softmax trên mảng rỗng.")
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def dot_product_attention(q, K, V):
    """Attention không scaling. Trả về scores, weights, context."""
    q, K, V = (np.asarray(a, dtype=np.float64) for a in (q, K, V))
    if q.ndim != 1 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("q phải là vector; K và V phải là ma trận.")
    if q.shape[0] != K.shape[1]:
        raise ValueError("Chiều của q phải khớp số cột của K.")
    if K.shape[0] != V.shape[0]:
        raise ValueError("K và V phải có cùng số hàng (số token).")
    scores = q @ K.T
    weights = softmax_stable(scores, axis=-1)
    return scores, weights, weights @ V


def scaled_dot_product_attention(Q, K, V):
    """Scaled attention: softmax(QK^T/sqrt(d_k))V.

    Q: (n_query, d_k), K: (n_key, d_k), V: (n_key, d_v).
    Trả về scores đã scale, attention weights và context.
    """
    Q, K, V = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q, K, V phải là ma trận 2 chiều.")
    if Q.shape[1] != K.shape[1]:
        raise ValueError("Q và K phải có cùng số chiều đặc trưng.")
    if K.shape[0] != V.shape[0]:
        raise ValueError("K và V phải có cùng số token.")
    if K.shape[1] == 0:
        raise ValueError("d_k phải lớn hơn 0.")
    scores = (Q @ K.T) / np.sqrt(K.shape[1])
    weights = softmax_stable(scores, axis=-1)
    return scores, weights, weights @ V


def save_rows_csv(path, rows):
    """Lưu danh sách dict thành CSV UTF-8; trả về đường dẫn đã ghi."""
    rows = list(rows)
    if not rows:
        raise ValueError("Cần ít nhất một dòng để tạo CSV.")
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_token_attention(tokens, weights, title, filename):
    """Vẽ biểu đồ cột attention theo token và lưu ảnh."""
    import matplotlib.pyplot as plt

    weights = np.asarray(weights, dtype=float).reshape(-1)
    if len(tokens) != len(weights):
        raise ValueError("Số token phải bằng số attention weights.")
    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.7), 4.5))
    ax.bar(range(len(tokens)), weights)
    ax.set_xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    ax.set_ylabel("Attention weight")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    return Path(filename)


def plot_attention_matrix(tokens, weights, title, filename):
    """Vẽ và lưu heatmap T×T; hàng là query token, cột là key token."""
    import matplotlib.pyplot as plt

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(tokens), len(tokens)):
        raise ValueError("Cần ma trận T×T khớp số token.")
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.65), max(5, len(tokens) * 0.55)))
    image = ax.imshow(weights, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tokens)), tokens)
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    return Path(filename)



def build_attention_classifier(vocab_size=20_000, embedding_dim=64,
                               max_length=200, dropout_rate=0.3,
                               learning_rate=1e-3):
    """Tạo Keras self-attention sentiment classifier và model lấy weights.

    TensorFlow được import tại chỗ để các hàm NumPy của module không phụ thuộc
    TensorFlow khi chỉ cần import attention_numpy.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    class ScaledSelfAttention(layers.Layer):
        def __init__(self, dimension, **kwargs):
            super().__init__(**kwargs)
            self.supports_masking = True
            self.dimension = int(dimension)
            self.query_projection = layers.Dense(self.dimension, use_bias=False)
            self.key_projection = layers.Dense(self.dimension, use_bias=False)
            self.value_projection = layers.Dense(self.dimension, use_bias=False)

        def call(self, inputs, mask=None):
            query = self.query_projection(inputs)
            key = self.key_projection(inputs)
            value = self.value_projection(inputs)
            scores = tf.matmul(query, key, transpose_b=True)
            scores = scores / tf.sqrt(tf.cast(self.dimension, scores.dtype))
            if mask is not None:
                key_mask = tf.cast(mask[:, tf.newaxis, :], tf.bool)
                scores = tf.where(key_mask, scores, tf.cast(-1e9, scores.dtype))
            weights = tf.nn.softmax(scores, axis=-1)
            context = tf.matmul(weights, value)
            return context, weights

        def compute_mask(self, inputs, mask=None):
            return mask, None

    class MaskedMeanPooling(layers.Layer):
        def call(self, inputs):
            context, token_ids = inputs
            valid = tf.cast(tf.not_equal(token_ids, 0)[..., tf.newaxis], context.dtype)
            total = tf.reduce_sum(context * valid, axis=1)
            count = tf.reduce_sum(valid, axis=1)
            return total / tf.maximum(count, tf.cast(1.0, context.dtype))

    token_ids = layers.Input(shape=(max_length,), dtype="int32", name="token_ids")
    embedded = layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim,
        mask_zero=True, name="embedding"
    )(token_ids)
    attention_layer = ScaledSelfAttention(embedding_dim, name="self_attention")
    context_sequence, attention_weights = attention_layer(embedded)
    pooled = MaskedMeanPooling(name="masked_mean_pooling")(
        [context_sequence, token_ids]
    )
    dropped = layers.Dropout(dropout_rate, name="dropout")(pooled)
    probability = layers.Dense(1, activation="sigmoid", name="sentiment")(dropped)

    model = models.Model(token_ids, probability, name="self_attention_classifier")
    weights_model = models.Model(token_ids, attention_weights, name="attention_weights")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, weights_model


def train_evaluate_attention_classifier(
    model, X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size=128, epochs=5, seed=42, verbose=1,
):
    """Huấn luyện rồi đánh giá classifier; trả model, dự đoán và metrics."""
    import time
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score

    tf.keras.utils.set_random_seed(seed)
    started = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
    )
    training_seconds = time.perf_counter() - started
    probabilities = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype("int32")
    labels = np.asarray(y_test).reshape(-1).astype("int32")
    return {
        "model": model,
        "history": history,
        "probabilities": probabilities,
        "predictions": predictions,
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="binary", zero_division=0)),
        "f1_type": "binary",
        "training_time_seconds": float(training_seconds),
        "parameters": int(model.count_params()),
    }
