import logging
import pickle

import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.config import MAX_LENGTH_PATH, MODEL_PATH, TOKENIZER_PATH

logger = logging.getLogger(__name__)


class CaptionService:
    """Loads trained model artifacts and generates captions for images."""

    def __init__(self):
        self.caption_model = load_model(MODEL_PATH)

        with open(TOKENIZER_PATH, "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(MAX_LENGTH_PATH, "rb") as f:
            self.max_length = pickle.load(f)

        self.vocab_size = len(self.tokenizer.word_index) + 1

        # Reverse lookup dict instead of linear scan per call
        self.index_to_word = {v: k for k, v in self.tokenizer.word_index.items()}

        inception = InceptionV3(weights="imagenet")
        self.feature_model = Model(
            inputs=inception.inputs,
            outputs=inception.get_layer("mixed10").output,
        )

    def _idx_to_word(self, integer: int) -> str | None:
        return self.index_to_word.get(integer)

    def extract_features(self, image_path: str) -> np.ndarray:
        image = load_img(image_path, target_size=(299, 299))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = self.feature_model.predict(image, verbose=0)
        feature = feature.reshape((1, 64, 2048))
        return feature

    def predict_greedy(self, feature: np.ndarray) -> str:
        in_text = "startseq"
        for _ in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length, padding="post")
            yhat = self.caption_model.predict([feature, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self._idx_to_word(yhat)
            if word is None:
                break
            in_text += " " + word
            if word == "endseq":
                break
        return in_text.replace("startseq ", "").replace(" endseq", "").strip()

    def predict_beam(self, feature: np.ndarray, beam_width: int = 3) -> str:
        sequences = [["startseq", 1.0]]

        for _ in range(self.max_length):
            all_candidates = []
            for seq in sequences:
                text, score = seq
                if text.split()[-1] == "endseq":
                    all_candidates.append(seq)
                    continue
                sequence = self.tokenizer.texts_to_sequences([text])[0]
                sequence = pad_sequences(
                    [sequence], maxlen=self.max_length, padding="post"
                )
                yhat = self.caption_model.predict([feature, sequence], verbose=0)[0]
                top_indices = np.argsort(yhat)[-beam_width:]
                for idx in top_indices:
                    word = self._idx_to_word(idx)
                    if word is None:
                        continue
                    new_score = score * yhat[idx]
                    all_candidates.append([text + " " + word, new_score])
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[
                :beam_width
            ]

        if not sequences:
            return ""
        best = sequences[0][0]
        return best.replace("startseq ", "").replace(" endseq", "").strip()

    def generate(
        self, image_path: str, method: str = "beam", beam_width: int = 3
    ) -> str:
        feature = self.extract_features(image_path)
        if method == "greedy":
            return self.predict_greedy(feature)
        return self.predict_beam(feature, beam_width=beam_width)
