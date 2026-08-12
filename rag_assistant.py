# Updated rag_assistant.py (only change is to return reply and conversation_id in process_query)

import sys

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
import os

os.environ["HF_HUB_OFFLINE"] = "1"
import re
import fitz
import requests
import time
import sqlite3
from llama_index.core import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import difflib
import traceback

RAW_INDEX_FILE = os.environ.get("RAW_INDEX_FILE", "raw_chunks.index")
RAW_REFS_FILE = os.environ.get("RAW_REFS_FILE", "raw_refs.pkl")
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://localhost:11434")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "llama3:8b-instruct-q4_0")


class RAGAssistant:
    def __init__(self, pdf_folder, db_path="assistant.db"):
        self.pdf_folder = pdf_folder
        self.db_path = db_path
        self.docs = []
        self.raw_texts = []
        self.meta_index = None
        self.meta_refs = []
        self.meta_embedder = None
        self.raw_index = None
        self.raw_refs = []
        self.field_map = {
            "موضوع": "subject",
            "حاضرین": "attendees",
            "حاضران": "attendees",
            "غایبین": "absentees",
            "غایب": "absentees",
            "غابیین": "absentees",
            "غایببن": "absentees",
            "غایبیین": "absentees",
            "تاریخ": "date",
            "رئیس جلسه": "boss",
            "رئیس": "boss",
            "رئيس": "boss",
            "دبیر جلسه": "secretary",
            "دبیر": "secretary",
            "دبير": "secretary",
            "ساعت شروع": "start_time",
            "ساعت جلسه": "start_time",
            "شروع جلسه": "start_time",
            "ساعت شرو": "start_time",
            "ساعت پایان": "end_time",
            "پایان جلسه": "end_time",
            "پایان": "end_time",
            "پايان": "end_time",
            "شماره": "number",
            "پیوست": "attachment",
            "خلاصه": "summary",
            "مصوبه": "resolutions",
            "مصوبات": "resolutions",
            "مصوبات جلسه جاری": "resolutions",
            "مسئول": "resolutions",
            "مسئول اجرا": "resolutions",
            "تاریخ اجرا": "resolutions",
            "مسئول پیگیری": "resolutions"
        }
        self.init_db()
        self.load_or_build_indexes()

    def init_db(self):
        """Initialize SQLite DB for conversations and searches, with migration for old schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(searches)")
        columns = [info[1] for info in cursor.fetchall()]
        old_schema = 'user_id' in columns and 'conversation_id' not in columns

        if old_schema:
            print("[*] Migrating old database schema...")
            cursor.execute('''CREATE TABLE IF NOT EXISTS conversations
                             (user_id TEXT, conversation_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS searches_new
                             (conversation_id INTEGER, query TEXT, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                              FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id))''')

            cursor.execute("SELECT DISTINCT user_id FROM searches")
            user_ids = [row[0] for row in cursor.fetchall()]

            for user_id in user_ids:
                cursor.execute("SELECT query, response, timestamp FROM searches WHERE user_id = ?", (user_id,))
                searches = cursor.fetchall()
                for query, response, timestamp in searches:
                    cursor.execute("INSERT INTO conversations (user_id, title, timestamp) VALUES (?, ?, ?)",
                                   (user_id, query[:50], timestamp))
                    conversation_id = cursor.lastrowid
                    cursor.execute(
                        "INSERT INTO searches_new (conversation_id, query, response, timestamp) VALUES (?, ?, ?, ?)",
                        (conversation_id, query, response, timestamp))

            cursor.execute("DROP TABLE searches")
            cursor.execute("ALTER TABLE searches_new RENAME TO searches")
            conn.commit()
            print("[*] Database migration completed.")
        else:
            cursor.execute('''CREATE TABLE IF NOT EXISTS conversations
                             (user_id TEXT, conversation_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS searches
                             (conversation_id INTEGER, query TEXT, response TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                              FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id))''')
            conn.commit()

        conn.close()

    def clean_persian_text(self, text):
        text = self.normalize_persian_text(text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_persian_text(self, text):
        replacements = {
            "ي": "ی", "ك": "ک",
            "ۀ": "ه", "ة": "ه",
            "ؤ": "و", "إ": "ا", "أ": "ا", "آ": "ا",
            "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
            "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.strip()

    def extract_metadata_from_pdfs(self):
        documents = []
        raw_texts = []

        for filename in os.listdir(self.pdf_folder):
            if not filename.lower().endswith(".pdf"):
                continue

            filepath = os.path.join(self.pdf_folder, filename)
            try:
                doc = fitz.open(filepath)
                full_text = ""
                for page in doc:
                    #"text" is just for the plain text and nothing else
                    full_text += page.get_text("text") + "\n"

                full_text = self.normalize_persian_text(full_text)
                raw_texts.append((filename, full_text))

                metadata = {"filename": filename}
                fields_order = [
                    ("date", "تاریخ"),
                    ("subject", "موضوع جلسه"),
                    ("boss", "رئیس جلسه"),
                    ("secretary", "دبیر جلسه"),
                    ("number", "شماره"),
                    ("attachment", "پیوست"),
                    ("start_time", "ساعت شروع"),
                    ("end_time", "ساعت پایان"),
                    ("attendees", "حاضرین"),
                    ("absentees", "غایبین"),
                    ("summary", "دستور کار/شرح خلاصه جلسه"),
                    ("resolutions", "مصوبات جلسه جاری")
                ]

                for i, (field_key, heading) in enumerate(fields_order):
                    normalized_heading = self.normalize_persian_text(heading)
                    next_headings = [re.escape(self.normalize_persian_text(h)) for _, h in fields_order[i + 1:]]
                    if field_key == "summary":
                        next_title = self.normalize_persian_text("مصوبات جلسه جاری")
                        pattern = rf"{re.escape(normalized_heading)}\s*(.*?)(?=(?:{re.escape(next_title)}|\n{{2,}}|\Z))"
                    elif field_key == "resolutions":
                        start_marker = self.normalize_persian_text("مسئول پیگیری")
                        end_marker = self.normalize_persian_text("نام و امضا حاضرین جلسه")
                        start_idx = full_text.find(start_marker)
                        end_idx = full_text.find(end_marker) if full_text.find(end_marker) != -1 else len(full_text)
                        value = full_text[start_idx + len(start_marker):end_idx].strip() if start_idx != -1 else ""
                    else:
                        pattern = rf"{normalized_heading}\s*[:：]?\s*(.*?)(?={'|'.join(next_headings)}|$)"

                    print(f"[DEBUG] Searching for heading '{normalized_heading}' in {filename}")
                    if field_key != "resolutions":
                        match = re.search(pattern, full_text, re.DOTALL)
                        if match:
                            value = self.normalize_persian_text(match.group(1).strip())
                            if field_key == "summary" and not value.strip():
                                metadata[field_key] = "خالی"
                                print(
                                    f"[DEBUG] Found heading '{normalized_heading}' in {filename}, but content is empty")
                            else:
                                metadata[field_key] = value if value.strip() else "خالی"
                                print(
                                    f"[DEBUG] Found heading '{normalized_heading}' in {filename}, value: {value[:50]}...")
                        else:
                            metadata[field_key] = "غایب"
                            print(f"[DEBUG] Heading '{normalized_heading}' not found in {filename}")
                    else:
                        if value:
                            metadata[field_key] = value if value.strip() else "خالی"
                            print(f"[DEBUG] Found heading '{normalized_heading}' in {filename}, value: {value[:50]}...")
                        else:
                            metadata[field_key] = "غایب"
                            print(f"[DEBUG] Heading '{normalized_heading}' not found in {filename}")

                documents.append(Document(text="", metadata=metadata))

            except Exception as e:
                print(f"[!] Error reading {filename}: {e}")

        self.docs = documents
        self.raw_texts = raw_texts
        return documents, raw_texts

    def build_metadata_index(self):
        embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        texts = []
        meta_refs = []
        for doc in self.docs:
            for key, value in doc.metadata.items():
                if key != "filename":
                    entry_text = f"{key}: {value}"
                    texts.append(entry_text)
                    meta_refs.append((doc.metadata["filename"], key, value))
        embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self.meta_index = index
        self.meta_refs = meta_refs
        self.meta_embedder = embedder
        return index, meta_refs, embedder

    def build_rawtext_index(self, chunk_size=200):
        texts = []
        refs = []
        for filename, full_text in self.raw_texts:
            words = full_text.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    texts.append(chunk)
                    refs.append((filename, chunk))
        embeddings = self.meta_embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index, refs

    def load_or_build_indexes(self):
        print("[*] Reading PDFs...")
        self.extract_metadata_from_pdfs()
        if not self.docs:
            print("[!] No valid PDFs found.")
            return

        print(f"[*] Loaded {len(self.docs)} document(s).")

        print("[*] Building metadata index...")
        self.build_metadata_index()

        if os.path.exists(RAW_INDEX_FILE) and os.path.exists(RAW_REFS_FILE):
            print("[*] Loading persistent raw text index...")
            self.raw_index = faiss.read_index(RAW_INDEX_FILE)
            with open(RAW_REFS_FILE, "rb") as f:
                self.raw_refs = pickle.load(f)
        else:
            print("[*] Building raw text index...")
            self.raw_index, self.raw_refs = self.build_rawtext_index()
            print("[*] Saving raw text index to disk...")
            faiss.write_index(self.raw_index, RAW_INDEX_FILE)
            with open(RAW_REFS_FILE, "wb") as f:
                pickle.dump(self.raw_refs, f)

        keywords = list(self.field_map.keys())
        self.keywords = keywords
        normalized_keywords = [self.normalize_persian_text(kw) for kw in keywords]
        self.normalized_keywords = normalized_keywords
        self.keyword_embeddings = self.meta_embedder.encode(normalized_keywords, convert_to_numpy=True,
                                                            normalize_embeddings=True)

    def retrieve_rawtext(self, query, top_k):
        q_emb = self.meta_embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.raw_index.search(q_emb, top_k)
        results = []
        for idx in idxs[0]:
            filename, chunk = self.raw_refs[idx]
            results.append(f"{filename} | {chunk}")
        return results

    def ask_llama3(self, prompt):
        try:
            response = requests.post(
                f"{LLM_ENDPOINT}/api/generate",
                json={
                    "model": LLM_MODEL_NAME,
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "[!] No response from the language model")
        except requests.RequestException as e:
            return f"[!] Error calling the language model: {e}"

    def process_query(self, query, user_id, user_name="", user_unit="", conversation_id=None, top_k_raw=20):
        try:
            matched_fields = []
            persian_labels = []
            normalized_query = self.normalize_persian_text(query)
            cleaned_query = self.clean_persian_text(query)
            query_words = normalized_query.split()

            print(f"[DEBUG] Normalized query: {normalized_query}")

            for keyword, field_name in self.field_map.items():
                normalized_keyword = self.normalize_persian_text(keyword)
                if normalized_keyword in normalized_query:
                    if field_name not in matched_fields:
                        matched_fields.append(field_name)
                        persian_labels.append(keyword)
                        print(f"[DEBUG] Exact match: {keyword} -> {field_name}")

            fuzzy_matches = []
            ngrams = []
            for n in range(1, min(4, len(query_words) + 1)):
                for i in range(len(query_words) - n + 1):
                    ngram = " ".join(query_words[i:i + n])
                    if len(ngram) >= 3:
                        ngrams.append(ngram)

            for keyword, field_name in self.field_map.items():
                normalized_keyword = self.normalize_persian_text(keyword)
                for ngram in ngrams:
                    threshold = 0.9 if len(normalized_keyword) < 5 else 0.85
                    similarity = difflib.SequenceMatcher(None, normalized_keyword, ngram).ratio()
                    print(f"[DEBUG] Fuzzy attempt: '{ngram}' vs '{keyword}' = {similarity:.2f}")
                    if similarity > threshold:
                        fuzzy_matches.append((field_name, keyword, similarity, ngram))

            seen_fields = set(matched_fields)
            for field_name, keyword, similarity, ngram in sorted(fuzzy_matches, key=lambda x: x[2], reverse=True):
                if field_name not in seen_fields:
                    seen_fields.add(field_name)
                    matched_fields.append(field_name)
                    persian_labels.append(keyword)
                    print(
                        f"[DEBUG] Fuzzy match: {keyword} -> {field_name} (similarity: {similarity:.2f}, matched n-gram: {ngram})")

            ngrams = []
            for n in range(2, min(4, len(query_words) + 1)):
                for i in range(len(query_words) - n + 1):
                    ngram = " ".join(query_words[i:i + n])
                    if len(ngram) >= 3:
                        ngrams.append(ngram)
            if not ngrams:
                for ngram in query_words:
                    if len(ngram) >= 3:
                        ngrams.append(ngram)

            semantic_matches = []
            for ngram in ngrams:
                ngram_embedding = self.meta_embedder.encode([ngram], convert_to_numpy=True, normalize_embeddings=True)
                similarities = np.dot(self.keyword_embeddings, ngram_embedding.T).flatten()
                max_idx = np.argmax(similarities)
                threshold = 0.9 if len(self.normalized_keywords[max_idx]) < 5 else 0.85
                if similarities[max_idx] > threshold:
                    keyword = self.keywords[max_idx]
                    field_name = self.field_map[keyword]
                    semantic_matches.append((field_name, keyword, similarities[max_idx], ngram))

            seen_fields = set(matched_fields)
            for field_name, keyword, similarity, ngram in sorted(semantic_matches, key=lambda x: x[2], reverse=True):
                if field_name not in seen_fields:
                    seen_fields.add(field_name)
                    matched_fields.append(field_name)
                    persian_labels.append(keyword)
                    print(
                        f"[DEBUG] Semantic match: {keyword} -> {field_name} (similarity: {similarity:.2f}, matched n-gram: {ngram})")

            if matched_fields:
                print(
                    f"[DEBUG] All matched fields: {', '.join(f'{label} ({field})' for label, field in zip(persian_labels, matched_fields))}")
            else:
                print("[DEBUG] No keywords matched, falling back to raw text search")

            relevant_docs = []

            if matched_fields:
                for doc in self.docs:
                    parts = [f"تاریخ: {doc.metadata.get('date', '')}"]
                    for field, label in zip(matched_fields, persian_labels):
                        value = doc.metadata.get(field, "غایب")
                        if field == "resolutions" and value != "غایب" and value != "خالی":
                            parts.append(f"{label}: {value}")
                        else:
                            parts.append(f"{label}: {value}")
                    relevant_docs.append("\n".join(parts))
            else:
                relevant_meta = self.retrieve_rawtext(query, top_k=20)
                seen_files = set()
                for match in relevant_meta:
                    filename = match.split(" | ")[0]
                    if filename not in seen_files:
                        seen_files.add(filename)
                        relevant_docs.append(match)

            raw_text_results = self.retrieve_rawtext(normalized_query, top_k=top_k_raw)
            combined_content = "\n\n".join(relevant_docs) if relevant_docs else ""
            if raw_text_results:
                total_length = len(combined_content)
                max_tokens = 1700
                char_limit = max_tokens * 4
                if total_length < char_limit:
                    remaining_chars = char_limit - total_length
                    raw_text_content = "\n\n".join(raw_text_results)
                    if len(raw_text_content) <= remaining_chars:
                        combined_content += "\n\n" + raw_text_content
                    else:
                        combined_content += "\n\n" + raw_text_content[:remaining_chars]

            personalized_query = f"{query} (Tailored for {user_name} in {user_unit})" if user_name and user_unit else query
            structured_prompt = (
                f"پرسش: {personalized_query}\n"
                f"اطلاعات مرتبط:\n"
                f"{combined_content}\n"
                f"لطفاً فقط بر اساس اطلاعات بالا پاسخ کامل و دقیق به فارسی بده."
            )

            print("\n[i] Context sent to the language model:")
            print(structured_prompt)

            print("\n[i] Response from the language model:")
            start_time = time.time()
            llm_reply = self.ask_llama3(structured_prompt)
            print(f"\nLLM response time: {time.time() - start_time:.2f} seconds")

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if conversation_id is None:
                cursor.execute("INSERT INTO conversations (user_id, title) VALUES (?, ?)", (user_id, query[:50]))
                conversation_id = cursor.lastrowid
            cursor.execute("INSERT INTO searches (conversation_id, query, response) VALUES (?, ?, ?)",
                           (conversation_id, query, llm_reply))
            conn.commit()
            conn.close()

            return llm_reply, conversation_id

        except Exception as e:
            print(f"[!] Error: {e}")
            traceback.print_exc()
            return "[!] An error occurred.", conversation_id

    def get_conversations(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT conversation_id, title, timestamp FROM conversations WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,))
        conversations = cursor.fetchall()
        conn.close()
        return conversations

    def get_conversation_searches(self, conversation_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT query, response, timestamp FROM searches WHERE conversation_id = ? ORDER BY timestamp ASC",
            (conversation_id,))
        searches = cursor.fetchall()
        conn.close()
        return searches

    def get_history(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT searches.query, searches.response, searches.timestamp 
            FROM searches 
            JOIN conversations ON searches.conversation_id = conversations.conversation_id 
            WHERE conversations.user_id = ? 
            ORDER BY searches.timestamp DESC
        """, (user_id,))
        history = cursor.fetchall()
        conn.close()
        return history

    def get_common_searches(self, user_id, top_n=20):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT searches.query, COUNT(*) as count 
            FROM searches 
            JOIN conversations ON searches.conversation_id = conversations.conversation_id 
            WHERE conversations.user_id = ? 
            GROUP BY searches.query 
            ORDER BY count DESC 
            LIMIT ?
        """, (user_id, top_n))
        commons = cursor.fetchall()
        conn.close()
        return commons