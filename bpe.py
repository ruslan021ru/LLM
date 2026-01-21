class BPE():
    "Токенизатор"

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._text_in_tokens = []
        self._uniq_tokens = []
        self.id2token = {}
        self.token2id = {}

    def fit(self, text: str) -> None:
        "Обучение токенизатора"
        self._uniq_tokens = sorted(list(set(text)))
        self._text_in_tokens = list(text)

        while len(self._uniq_tokens) < self.vocab_size:
            token = self._get_popular_pair_token()
            self._tokens_consolidation(token)
            self._uniq_tokens.append(token)

        for ind, token in enumerate(self._uniq_tokens):
            self.id2token[ind] = token
            self.token2id[token] = ind

    def encode(self, text: str) -> list[int]:
        "Энкодинг"

        encode_id = list()
        i = 0
        while i < len(text):
            tokens_starting_with_char = self._get_tokens_start_char(text[i])
            longest_token, len_token = self._get_longest_token(text, i, tokens_starting_with_char)
            
            encode_id.append(self.token2id[longest_token])
            i += len_token

        return encode_id

    def _get_tokens_start_char(self, char: chr) -> list[str]:
        "Извлечение всех токенов, начинающихся с заданного символа"
        tokens = list()
        for token in self.token2id:
            if token[0] == char:
                tokens.append(token)
        return tokens

    def _get_longest_token(self, text: str, i: int, tokens: list[str]) -> tuple[str, int]:
        "Извлечение самого длинного токена из текста"
        max_len = 0
        longest_token = ""
        for token in tokens:
            len_token = len(token)
            if token == text[i:i+len_token] and len_token > max_len:
                longest_token = token
                max_len = len_token

        return longest_token, max_len


    def decode(self, tokens_ids: list[int]) -> str:
        "Декодинг"
        text = ""
        for id_token in tokens_ids:
            text += self.id2token[id_token]

        return text

    def _get_popular_pair_token(self) -> str:
        "Получение самой часто встречаемой пары последовательных токенов"
        token_pair_frequency = dict()
        for i in range(len(self._text_in_tokens)-1):
            token_pair = self._text_in_tokens[i] + self._text_in_tokens[i+1]
            token_pair_frequency[token_pair] = token_pair_frequency.get(token_pair, 0) + 1

        return max(token_pair_frequency, key=token_pair_frequency.get)

    def _tokens_consolidation(self, token: str) -> None:
        "Объединение самой часто встречаемой пары токенов"
        new_text_in_tokens = list()
        skip_flag = False
        for i in range(len(self._text_in_tokens)-1):
            token_pair = self._text_in_tokens[i] + self._text_in_tokens[i+1]
            if token_pair == token:
                new_text_in_tokens.append(token_pair)
                skip_flag = True
            elif skip_flag: skip_flag = False
            else: new_text_in_tokens.append(self._text_in_tokens[i])

        if not skip_flag: new_text_in_tokens.append(self._text_in_tokens[-1])

        self._text_in_tokens = new_text_in_tokens

    def save(self, filename: str) -> None:
        "Сохранение токенизатора"
        with open(filename, 'wb') as f:
            dill.dump(self, f)
        print(f"Объект сохранён в {filename}")

    @classmethod
    def load(cls, filename: str) -> "BPE":
        "Загрузка токенизатора"
        with open(filename, 'rb') as f:
            obj = dill.load(f)

        print(f"Объект загружен из {filename}")
        return obj
    