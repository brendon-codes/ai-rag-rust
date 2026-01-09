# SIMPLE AI RAG EXAMPLE

## SETUP

```sh
uv sync
cp .env.example .env
```

## DEVELOPMENT

Create pinecone index

```sh
uv run -m rag.bin.make_db
```

Create embeddings

```sh
uv run -m rag.bin.make_embeddings
```

Upload to Pinecone

```sh
uv run -m rag.bin.upload_embeddings
```

Query Pinecone For Similarity Matches

```sh
echo "tested on animals?" | uv run -m rag.bin.find_similar
```

Code quality checks

```sh
uv run ruff check
uv run basedpyright
```

## END USER USAGE

Get Full Agent Response to Question

```sh
echo "tested on animals?" | uv run -m rag.bin.ask
echo "main ingredient? tell me in spanish" | uv run -m rag.bin.ask
```
