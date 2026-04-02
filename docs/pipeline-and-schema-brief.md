# Pipeline & DB Schema (Brief)

This doc explains how intent search works today, what is embedded, and how `intents` differ from `utterances`.

## 1) Big picture pipeline

### A) Offline / data management side
- You create/update intents via `/api/v1/intents`.
- Each intent has:
  - `intent_code` (label/key),
  - `description` (human text),
  - `is_active`.
- When an intent is created/updated, backend also creates or updates a **canonical utterance** (`source = intent_description`) and embeds that text.
- You can add more examples via `/api/v1/intents/{intent_id}/utterances`; each utterance is embedded too.

### B) Online inference side (`/api/v1/inference/intent`)
1. Receive audio file (multipart).
2. STT transcribes audio -> transcript text.
3. Embedding model converts transcript text -> vector.
4. Similarity search compares this vector against stored utterance vectors.
5. Results are aggregated per intent and confidence policy decides matched vs low_confidence.

## 2) What is searched for similarity?

The system compares the query vector to rows in `intent_embeddings.embedding` using pgvector cosine distance:

- Score expression in code:
  - `score = 1 - cosine_distance(stored_embedding, query_embedding)`
- Higher score = more similar.

Then it:
- filters by `Intent.is_active = true`
- filters by `IntentUtterance.language_code == language_hint`
- groups by intent
- takes `max(score)` per intent (best matching utterance for that intent)
- orders descending and returns top-k.

So similarity is **utterance-level matching**, then **intent-level aggregation**.

## 3) Intent vs Utterance (important distinction)

### Intent (`intents` table)
- Represents the business class/label (e.g. `balance_inquiry`).
- Has metadata (`description`, active flag).
- Not directly what search compares vectors against.

### Utterance (`intent_utterances` table)
- Example phrase tied to an intent.
- Multiple utterances can belong to the same intent (1 intent -> many utterances).
- This is the primary text unit that gets embedded and searched.

### Embedding (`intent_embeddings` table)
- Stores vector per utterance + model name.
- FK is `utterance_id -> intent_utterances.id`.
- Unique per (`utterance_id`, `model_name`).

## 4) Can intents have multiple embedded fields?

Current implementation: **effectively yes, through utterances**.

- An intent can have many utterances.
- Each utterance has one embedded text field (`text`).
- Search checks all those utterance embeddings, then maps back to the intent.

Current implementation does **not** embed separate intent fields like:
- `intent_code` as its own embedding,
- `description` as a separate embedding record independent of utterances,
- custom metadata fields (unless you materialize them as utterances).

If you want multi-field semantics, the existing pattern is to add synthetic utterances (for example one from description, one from synonyms, one from FAQ text).

## 5) Core schema map

- `intents`
  - PK: `id`
  - unique: `intent_code`
- `intent_utterances`
  - PK: `id`
  - FK: `intent_id -> intents.id` (cascade delete)
- `intent_embeddings`
  - PK: `id`
  - FK: `utterance_id -> intent_utterances.id` (cascade delete)
  - vector column: `embedding vector(1024)`
  - unique: (`utterance_id`, `model_name`)

Related inference logging tables:
- `inference_requests`
- `inference_results`
- `model_registry`

## 6) Practical takeaway

- Think of `intents` as classes.
- Think of `utterances` as training/index examples for those classes.
- Search quality is mostly driven by utterance coverage/quality, language correctness, and embedding model quality.
