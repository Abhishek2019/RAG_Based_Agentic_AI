import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, re, glob, json
import ollama
from pymilvus import MilvusClient
from pypdf import PdfReader
import numpy as np
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# ---------- Config ----------
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini:3.8b-fp16")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini:latest")
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "nomic-embed-text")
DB_PATH      = os.getenv("MILVUS_LITE_PATH", "./milvus.db")
COLLECTION   = os.getenv("MILVUS_COLLECTION", "docs")
CHUNK_CHARS  = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVER   = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K        = int(os.getenv("TOP_K", 3))


def read_pdf(path:str):

    '''
    Read input PDF file, pagewise, and return text :str
    '''
    
    out = []
    r = PdfReader(path)
    
    for page in r.pages:

        t = page.extract_text() or ""
        out.append(t)

    return "\n".join(out)


def load_docs(folder: str):

    '''
    Load the .pdf and .txt documents
    '''
    docs = []

    for path in glob.glob(os.path.join(index,"**","*"),recursive=True):

        if not os.path.isfile(path):

            continue

        ext = os.path.splitext(path)[1].lower().strip()

        try:
            if ext in [".pdf"]:
    
                text = read_pdf(path)
    
            elif ext in [".txt"]:
    
                continue
    
            else:
                continue
    
            if text.strip():
    
                docs.append({"doc_id": os.path.basename(path),
                            "text": text
                            })
        except Exception as e:
            print(e)
            print(f"[skip] {path}")
            
            
    return docs

            

        
    
def chunk_text(text: str, size=1200, overlap=200):

    '''
    Chunk the input text for the minimum given size (default: 1200 chars), with overlap (default: 200 chars) for the next
    '''

    sents = re.split(r'(?<=[\.\!\?])\s+', text.strip())

    chunks, cur = [], ""

    for s in sents:

        if len(s)+len(cur) <= size:

            cur = cur+(" " if cur else "") + s

        else:

            if cur:
                chunks.append(cur)

            tail = cur[-overlap:] if overlap >0 else ""

            cur = (tail + " " + s).strip()

    if cur:
        chunks.append(cur)

    return chunks



def embed_texts(texts: List[str]):
    '''
    Embed the input batch of chunks
    '''

    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    
    return np.array(resp["embeddings"], dtype=np.float32)



client = MilvusClient(DB_PATH)

def ensure_collection(dimension: int):

    collections_list = client.list_collections()
    print("collections_list : ", collections_list)
    
    if COLLECTION not in collections_list:

        
        client.create_collection(
            
            collection_name = COLLECTION,
            dimension = dimension,
            metric_type="COSINE",
            index_params = {"index_type": "AUTOINDEX"},
            auto_id=True
            
        )

        print(f"[milvus] created collection {COLLECTION} , dim={dimension}")
        
                

def index_folder(folder: str):

    '''
    Create chunks, embed them and push records with doc_id, chunk_id into Milvus(DB_PATH)

    TO DO: Prepare cluster (on Oracle Cloud Free Tier) for Milvus instead of .db file
    '''
    
    docs = load_docs(folder)

    if not docs:

        print("[index] no documents found")

    #---------Chunk & Embed Documents -------------------------------

    batch  = []
    meta_batch = []
    sample_emb_dim = None
    records = []
    
    for d in docs:

        chunks = chunk_text(d["text"], CHUNK_CHARS, CHUNK_OVER)
        
        for i, ch in enumerate(chunks):

            batch.append(ch)

            meta_batch.append({"doc_id":d["doc_id"],
                              "chunk_id": i,
                               "text": ch
                              })
            
            if len(batch) >= 64:

                vecs = embed_texts(batch)

                if sample_emb_dim is None:

                    sample_emb_dim = len(vecs[0])
                    ensure_collection(sample_emb_dim)

                for m,v in zip(meta_batch, vecs):

                    records.append({"vector":v, **m})
                
                batch, meta_batch = [], []
        
    if batch:
        vecs = embed_texts(batch)
        
        if not client.has_collection(COLLECTION):
            ensure_collection(len(vecs[0]))

        for m,v in zip(meta_batch, vecs):
            records.append({"vector":v, **m})
        
        batch, meta_batch = [], []

    if records:

        client.insert(collection_name=COLLECTION, data=records)
        print(f"records are inserted into collection: {COLLECTION}, records length: {len(records)}")


SYSTEM_1 = """
You are a helpful assistant.
If the user asks about people or resumes, you MUST call the tool `retrive_from_milvus`.


The output MUST be a valid JSON object only.
- Do NOT wrap it in triple backticks.
- Do NOT include ```json or any Markdown formatting.
- Do NOT add explanations, only return the JSON object.

The JSON format is:

{
  "type": "function",
  "function": {
    "name": "retrive_from_milvus",
    "arguments": { "query": "<user text>", "k": 3 }
  }
}

Do not answer directly. Always return only the JSON object when invoking the tool.

But if tool is not required or received input from tool answer in natural language.
"""

SYSTEM_2 = ''' You are a helpful research assistant.
- When answering user questions, use the information gathered from role: "tool".
- Cite each quote with {doc_id;chunk:id}.
- If nothing is found, say so and ask the user to add documents.
'''

def milvus_search(query: str, k: int=TOP_K):

    qvec = embed_texts([query])
    
    res = client.search(
        collection_name = COLLECTION,
        data=qvec,
        limit=k,
        output_fields = ["doc_id", "chunk_id", "text"]
    )

    hits = res[0] if res else []

    # Normalize to compact a summary for the LLM

    out = []

    for h in hits:

        out.append({
            "doc_id": h.get("entity", {}).get("doc_id"),
            "chunk_id": h.get("entity", {}).get("chunk_id"),
            "score": h.get("distance"),
            "text": h.get("entity", {}).get("text"),
        })

    return out

    

def retrive_from_milvus(query: str, k: int = TOP_K ):

    '''
    Retrive top k chunks from the milvus DB. Include score variable along with, doc_id, text and chunk_id
    '''

    return milvus_search(query, k)
    

def chat_once(question: str):

    '''
    '''

    messages =[
        {"role": "system", "content": SYSTEM_1},
        {"role": "user", "content": question}
    ]

    response = ollama.chat(
        model = OLLAMA_MODEL,
        messages = messages,
        tools = [{
            "type": "function",
            "function": {
                "name": "retrive_from_milvus",
                "description": "Search the Milvus database of resumes to answer questions about people.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The text to search for"},
                        "k": {"type": "integer", "description": "Number of results to return"}
                    },
                    "required": ["query"]
                }
            }
        }]
    )

    available = {"retrive_from_milvus": retrive_from_milvus}
    
    tool_calls = getattr(response.message,"tool_calls",None)

    if tool_calls:
        
        for call in tool_calls:
    
            fn = available.get(call.function.name)
    
            if fn:
                
                tool_out = fn(**call.function.arguments)

                messages = [m for m in messages if m["role"] not in ("system")]
                messages.append({"role": "system", "content": SYSTEM_2})
                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_out),
                    "name": call.function.name
                })

    else:
        content = getattr(response.message,"content", None)

        if content:

            content = json.loads(content)

            if content["type"] == "function":

                fn = available.get(content["function"]["name"])

                if fn:
                    tool_out = fn(**content["function"]["arguments"])

                    messages = [m for m in messages if m["role"] not in ("system")]
                    messages.append({"role": "system", "content": SYSTEM_2})
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_out),
                        "name": content["function"]["name"]
                    })
        
    final = ollama.chat(
        model= OLLAMA_MODEL,
        messages = messages
        
    )

    return final.message.content


    
# agent initialize

if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("--index",type= str,help="Path to folder of docs to ingest")
    ap.add_argument("--ask", type= str, help="ASk a question, to get an answer based on documents using LLM")

    args = ap.parse_args()

    if args.index:
        if COLLECTION not in client.list_collections():
            print(f"documents alredy exists at location {DB_PATH}")
            index_folder(args.index)
    if args.ask:
        output = chat_once(args.ask)
        console.print(Panel.fit(Text(output, style="white"), border_style="green"))


    

    
