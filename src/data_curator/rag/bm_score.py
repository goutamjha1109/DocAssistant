"""https://arpitbhayani.me/blogs/bm25/"""

from src.data_curator.chunking.tokenizer import encode, count_tokens, decode_single_tokens
import re
import math

class BM25Scorer:
    def __init__(self,query:str, chunks:list[str],k1:float=1.2,b:float=0.75):
        self.query = query
        self.chunks = chunks
        self.K1 = k1
        self.B = b
        self.IDF_MAP = {}
        self.avgdl = self.__get_average_token_length()
    
    def __get_average_token_length(self) -> float:
        return sum(count_tokens(chunk) for chunk in self.chunks)/len(self.chunks)

    @staticmethod
    def create_tokens(text:str) -> list[str]:
        query_tokens = encode(text)
        text_tokens = decode_single_tokens(query_tokens)
        return text_tokens
    
    def get_inverse_document_frequency_for_token(self, token:str) -> float:
        pattern = re.compile(pattern=rf"\b{re.escape(token)}\b",flags=re.IGNORECASE)
        found_in_chunks =  sum(1 for chunk in self.chunks if pattern.search(chunk))
        total_chunks = len(self.chunks)
    
        return math.log((total_chunks-found_in_chunks+0.5)/(found_in_chunks+0.5))
    
    def init_idf_map(self,tokens:list[str]):
        for token in tokens:
            if token not in self.IDF_MAP:
                idf = self.get_inverse_document_frequency_for_token(token)
                self.IDF_MAP[token] = idf
            else:
                print(f"Token: {token} is already in map, skipping IDF calculation...")
    
    @staticmethod        
    def term_frequency_per_chunk(token:str,chunk:str):
        pattern = re.compile(pattern=rf"\b{re.escape(token)}\b",flags=re.IGNORECASE)
        return len(pattern.findall(chunk))
        
    def get_bm25_score_per_chunk(self,chunk:str,tokens:list[str]):
        d = count_tokens(chunk)
        score = 0
        for token in tokens:
            idf = self.IDF_MAP.get(token,0)
            f = self.term_frequency_per_chunk(token,chunk)
            
            score += idf*((f*(self.K1+1))/(f+self.K1*(1-self.B+self.B*d/self.avgdl)))
        return score

    def get_bm25_scores(self):
        tokens = self.create_tokens(self.query)
        self.init_idf_map(tokens)
        
        CHUNK_MAP = []
        for idx,chunk in enumerate(self.chunks):
            score = self.get_bm25_score_per_chunk(chunk,tokens)
            CHUNK_MAP.append({'index':idx,
            'chunk_text':chunk,
            'score':score})
            print(f"Chunk index: {idx} has BM25 {score}")
        
        return CHUNK_MAP