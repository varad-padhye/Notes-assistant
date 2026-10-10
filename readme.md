# RAG-Powered Notes Assistant

A command-line question-answering assistant that answers questions from
textbooks supplied as PDFs. It retrieves relevant passages from a PostgreSQL
database, re-ranks them, and asks Gemini to generate a textbook-grounded
response with citations.

## What this project does

The project is a Retrieval-Augmented Generation (RAG) application with two main
workflows:

1. **Ingestion:** `ingest.py` extracts text and tables from PDFs, splits the
   content into overlapping chunks, creates embeddings, and stores each chunk
   and its page/source information in PostgreSQL with pgvector.
2. **Question answering:** `pipeline.py` embeds a question, retrieves the
   closest textbook chunks, re-ranks them with a cross-encoder, and sends the
   best excerpts to Gemini. The CLI displays the answer, citations, confidence,
   whether an answer was found, whether excerpts conflict, and any escalation
   note.

The assistant is instructed to use only retrieved excerpts, cite factual
claims, indicate when the excerpts do not support an answer, and flag
conflicting explanations. Gemini is asked for JSON output. If generation
returns malformed JSON or another generation error, the application retries
once before returning a refusal.

## Technology

- Python 3.10+
- PostgreSQL 16 with pgvector, run through Docker Compose
- `pdfplumber` for PDF text and table extraction
- `sentence-transformers/all-MiniLM-L6-v2` for 384-dimensional embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for re-ranking
- Gemini through the `google-genai` Python package

## Project files

| File | Purpose |
| --- | --- |
| `ingest.py` | Reads PDFs, chunks and embeds their content, and writes it to the database |
| `pipeline.py` | Retrieves and re-ranks chunks, calls Gemini, parses results, and runs the interactive CLI |
| `db.py` | Shared database connection and embedding-model helpers |
| `schema.sql` | Creates the pgvector extension, chunk table, and indexes |
| `docker-compose.yml` | Starts the PostgreSQL/pgvector database |
| `requirements.txt` | Python dependencies |
| `.env.example` | Example environment-variable values |
| `test_pipeline.py` | Regression test for retrying malformed Gemini responses |

## Setup and start

### 1. Prerequisites

Install:

- Python 3.10 or newer
- Docker Desktop with Docker Compose
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### 2. Create and activate a Python environment

From the project directory, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, either allow script activation for
your user according to your organization's policy, or run the project using the
environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Start PostgreSQL

```powershell
docker compose up -d
docker compose ps
```

The Compose configuration exposes PostgreSQL on `localhost:5432`, creates the
`policydb` database, and mounts `schema.sql` for initial database setup. The
default database settings are:

| Setting | Default |
| --- | --- |
| Host | `localhost` |
| Port | `5432` |
| Database | `policydb` |
| User | `postgres` |
| Password | `password` |

These defaults are intended for local development. Set `PG_HOST`, `PG_PORT`,
`PG_DB`, `PG_USER`, and `PG_PASSWORD` in the environment before running the
scripts if your database uses different settings.

### 4. Add textbook PDFs

Put one or more text-based textbook PDFs in the `policies` folder at the project
root. The folder is created automatically if it does not exist when ingestion
is started.

The ingester uses each PDF's filename to derive its title and source ID. It
extracts text and tables page by page and records page numbers for citations.
Scanned image-only PDFs are not OCR'd and will not produce usable text.

### 5. Ingest the PDFs

To add or update chunks without clearing existing records:

```powershell
python ingest.py
```

To clear the table and rebuild it from the PDFs currently in the folder:

```powershell
python ingest.py --replace
```

To use a different PDF folder:

```powershell
python ingest.py --policies-dir "C:\path\to\textbook-pdfs"
```

Use `--replace` carefully: it truncates all existing textbook chunks before
ingesting the selected folder.

### 6. Configure the Gemini API key

The application looks for a key in this order:

1. `GEMINI_API_KEY` environment variable
2. `gemini_key.txt` in the project directory
3. Interactive prompt when `pipeline.py` starts

If you enter the key at the prompt, it is saved in `gemini_key.txt` for
subsequent runs. `.gitignore` excludes this file; do not commit or share it.

For a PowerShell session, you can set the key like this:

```powershell
$env:GEMINI_API_KEY = "your_gemini_api_key"
```

The model defaults to `gemini-3.8-flash`. To select a different model supported
by your Gemini account, set `GEMINI_MODEL`:

```powershell
$env:GEMINI_MODEL = "your-model-name"
```

Although `.env.example` documents the available settings, the Python scripts do
not automatically load `.env` files. Set environment variables in the shell,
or use the key prompt and the default database settings.

### 7. Ask questions

Start the interactive assistant:

```powershell
python pipeline.py
```

Enter a question at the prompt, for example:

```text
Ask a textbook question: What are parameters?
```

The response includes the answer status, confidence, conflict status, latency,
number of retrieved chunks, answer, and citations (when returned). Enter
`exit` or `quit` to close the assistant.

## How answering works

For each question, the application:

1. Creates a query embedding using `all-MiniLM-L6-v2`.
2. Retrieves up to 8 nearest chunks using pgvector cosine distance.
3. Re-ranks the results and uses up to 5 excerpts as Gemini context.
4. Requests a JSON response with answer, citations, confidence, and conflict
   information.
5. Parses the response and displays the result. If re-ranking fails, it uses
   the original retrieval order. If Gemini generation fails, it retries once;
   after a second failure it returns a refusal with an escalation note.

## Troubleshooting

### No PDF files found

Check that the PDFs are directly inside `policies` (or the directory passed to
`--policies-dir`) and that the files have a `.pdf` extension.

### No text could be extracted

The PDFs may contain scanned page images rather than selectable text. OCR is
not currently included; use text-based PDFs or OCR the documents before
ingesting them.

### Database connection or missing-table error

Check that Docker Desktop is running and the database container is healthy:

```powershell
docker compose ps
docker compose logs postgres
```

Confirm the `PG_*` environment variables match the database. The schema mount
in `docker-compose.yml` is applied when PostgreSQL initializes its data
directory; it does not automatically re-run `schema.sql` against an existing
database volume.

### Gemini API key or model error

Check that the key is valid and that `GEMINI_MODEL` names a model available to
your account. You can set `GEMINI_API_KEY` in the current PowerShell session or
remove/update the local `gemini_key.txt` so the program prompts again.

### Invalid JSON or incomplete answer

The application logs JSON parsing failures and retries once. If both responses
are invalid, it returns a refusal and an escalation note instead of displaying
an unparsed answer. Check the console logs for Gemini's reported finish reason
and ensure the selected model is supported.

## Run the regression test

With the project environment active:

```powershell
python -m unittest test_pipeline.py
```
