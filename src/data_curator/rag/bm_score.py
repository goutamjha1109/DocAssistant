"""https://arpitbhayani.me/blogs/bm25/"""

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
import math

_stemmer = SnowballStemmer(language="english")
_stopwords = set(map(_stemmer.stem,stopwords.words("english")))

class BM25Scorer:
    def __init__(self, query: str, chunks: list[dict[str,str]], k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 Scorer"""
        self.query = query
        self.chunks:list[dict] = chunks # Contains chunk_index, paper_id, chunk_text, title
        self.K1 = k1
        self.B = b
        self.IDF_MAP = {}
        self.cleaned_chunks:list[dict] = [
            {**chunk, "cleaned_chunk":" ".join(self.create_clean_tokens(chunk['chunk_text']))}
            for chunk in self.chunks
        ] # Clean every chunk for accurate tf and idf calculation
        self.avgdl = self.__get_average_token_length()

    def __get_average_token_length(self) -> float:
        """
        Calcuates average token length across the documents.

        Returns:
            float: calculates 'avgdl' parameter
        """
        return sum(len(chunk['cleaned_chunk'].split()) for chunk in self.cleaned_chunks) / len(self.cleaned_chunks)

    @staticmethod
    def create_clean_tokens(text: str) -> list[str]:
        """
        Tokenizes words by splitting on spaces, and cleaning every token
        Tokenizes both query and chunk documents using same method for exact matching

        Args:
            text (str): text to create tokens for

        Returns:
            list[str]: list of cleaned tokens
        """
        token_list = []
        for word in text.split():
            cleaned_word = re.sub(r"[^a-zA-Z0-9]","",word)
            if not cleaned_word: # Removing non alphanumeric words
                continue
            token = _stemmer.stem(cleaned_word) # Already lowers the word
            if token not in _stopwords:
                token_list.append(token)
        return token_list

    def get_inverse_document_frequency_for_token(self, token: str) -> float:
        """
        Calcuates inverse document frequency of a token wrto ingested chunks.
        Args:
            token (str): token to create idf for

        Returns:
            float: idf
        """
        print(f"Calculating IDF for query token: {token}")
        pattern = re.compile(pattern=rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
        found_in_chunks = sum(1 for cleaned_chunk in self.cleaned_chunks 
                              if pattern.search(cleaned_chunk['cleaned_chunk']))
        total_chunks = len(self.cleaned_chunks)

        idf = math.log(
            (total_chunks - found_in_chunks + 0.5) / (found_in_chunks + 0.5)
        )
        return max(0,idf) # Negative IDF scores are restricted to 0

    def init_idf_map(self, tokens: list[str]):
        """Initialize idf for all the query tokens

        Args:
            tokens (list[str]): query tokens
        """
        for token in tokens:
            if token not in self.IDF_MAP:
                idf = self.get_inverse_document_frequency_for_token(token)
                print(f"IDF for query token {token}: {idf}")
                self.IDF_MAP[token] = idf
            else:
                print(f"Token: {token} is already in map, skipping IDF calculation")

    @staticmethod
    def term_frequency_per_chunk(token: str, cleaned_chunk: str) -> int:
        """Calculate term frequency of a token in a document

        Args:
            token (str): query token
            cleaned_chunk (str): Cleaned document

        Returns:
            int: term frequency
        """
        pattern = re.compile(pattern=rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
        return len(pattern.findall(cleaned_chunk))

    def get_bm25_score_per_chunk(self, cleaned_chunk: str, tokens: list[str]) -> float:
        """
        Calculate BM25 Score for a document given a query.
        Calculates exact query token occurence.  

        Args:
            cleaned_chunk (str): Cleaned chunks
            tokens (list[str]): Query Tokens

        Returns:
            float: bm25 score
        """
        d = len(cleaned_chunk.split())
        score = 0
        for token in tokens:
            idf = self.IDF_MAP.get(token, 0)
            f = self.term_frequency_per_chunk(token, cleaned_chunk)

            score += idf * (
                (f * (self.K1 + 1))
                / (f + self.K1 * (1 - self.B + self.B * d / self.avgdl))
            )
        return score

    def get_bm25_scores(self,top_k = 20) -> list[dict]:
        """
        Calculate bm25 score for all the chunks and retrieves top k scored chunks for generation.

        Args:
            top_k (int): Top scored documents Defaults to 20.

        Returns:
            list[dict]: Top scored exact matched chunks for better generation
        """
        print(f"Starting BM25 Score calculations..")

        query_tokens = self.create_clean_tokens(self.query)
        print(f"Query:{self.query} converted to tokens {' '.join(query_tokens)}")

        self.init_idf_map(query_tokens)
        print(f"Initialized IDF Map for query tokens given the knowledge base.")

        scored_chunks = []
        for cleaned_chunk in self.cleaned_chunks:
            score = self.get_bm25_score_per_chunk(cleaned_chunk['cleaned_chunk'], query_tokens)
            print(f"Chunk index: {cleaned_chunk['chunk_index']} has BM25 {score}")
            cleaned_chunk.update({"score": score})
            scored_chunks.append(cleaned_chunk)

        return sorted(scored_chunks, key = lambda x: x['score'], reverse= True)[:top_k]


# if __name__ == "__main__":
#     from src.data_curator.chunking.splitter import recursive_character_split
#     path = r"D:\DocAssistant\data\processed\2606.20527v1.txt"
#     with open(path,'r',encoding='utf-8') as file:
#         text = file.read()
#         chunks = recursive_character_split(text)

#     question = "What is the definition of StylisticBias as introduced in the paper?"
#     scorer = BM25Scorer(query=question,chunks=chunks)
#     scored_chunks = scorer.get_bm25_scores()
#     top_5_chunks = sorted(scored_chunks, key=lambda x:x['score'],reverse=True)[:5]

#     for chunk in top_5_chunks:
#         print("="*10)
#         print(chunk['chunk_text'])
#         print(chunk['score'])
#         print("="*10)
