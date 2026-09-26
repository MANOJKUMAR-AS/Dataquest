CHATGPT :https://chatgpt.com/g/g-p-6ab73fcb43848191bd5253d0b2af7674-dataquest/project
GEMINI: https://share.gemini.google/e9rNehpdq2T3
https://share.gemini.google/NfEEYOVLmJii




# DataQuest – Context Lag Problem

## Overview

This project addresses the **Context Lag Problem** in conversational AI.

The goal is to identify situations where a conversational model may fail to correctly use earlier context when responding to a later turn in a conversation.

The project is divided into two tasks:

* **Task A – Context Lag Detection**
* **Task B – Context Similarity / Candidate Ranking**

We use NLP-based approaches including **Sentence Transformers** and **BM25** to identify relevant conversational context.

---

# Project Structure

```text
Dataquest/
│
├── data/
│   ├── taskA_hidden_public.csv
│   └── ...
│
├── outputs/
│   ├── submission_taskA.csv
│   ├── taskA_transcripts.csv
│   ├── taskA_transcripts.txt
│   └── ...
│
├── Task A/
│   └── ...
│
├── Task B/
│   └── ...
│
├── test_bm25.py
├── requirements.txt
└── README.md
```

---

# Task A – Context Lag Detection

## Objective

Task A focuses on identifying the correct previous conversational context for a given target utterance.

In a multi-turn conversation, a user's current statement may depend on information mentioned several turns earlier.

The objective is therefore to:

1. Understand the conversation.
2. Identify the relevant previous context.
3. Match the current utterance with its supporting context.
4. Produce the required prediction/submission format.

---

## Approach

For Task A, we used semantic similarity techniques based on **Sentence Transformers**.

Sentence Transformers convert text into dense vector representations called **embeddings**.

These embeddings allow us to compare the semantic meaning of two pieces of text rather than relying only on exact keyword matches.

### Pipeline

```text
Conversation
     │
     ▼
Extract candidate previous turns
     │
     ▼
Generate sentence embeddings
     │
     ▼
Calculate semantic similarity
     │
     ▼
Rank candidate contexts
     │
     ▼
Select relevant context
     │
     ▼
Generate Task A submission
```

---

## Models Used

We experimented with two Sentence Transformer models:

* Sentence Transformer Model 1
* Sentence Transformer Model 2

Both models were evaluated based on their ability to identify semantically relevant conversational context.

The models allow the system to capture relationships such as:

```text
Earlier:
"I booked a flight to Delhi for Monday."

Later:
"What time does my flight leave?"

```

Even though the two sentences do not share many exact words, their semantic relationship can be captured through embeddings.

---

## Why Sentence Transformers?

Traditional keyword matching can fail when two utterances use different words but express related meanings.

For example:

```text
"I purchased a laptop yesterday."

"I want to know its warranty period."
```

A keyword-based approach may struggle to connect these statements.

Sentence embeddings provide a semantic representation that makes it possible to identify the relationship between them.

---

# Task A Outputs

The project generates:

### Submission File

```text
outputs/submission_taskA.csv
```

This contains the predictions required for Task A evaluation.

### Judge-Friendly Transcripts

```text
outputs/taskA_transcripts.txt
outputs/taskA_transcripts.csv
```

These files organize the conversations into an easier-to-read format so that predictions can be manually inspected.

The transcript generation does **not modify the official submission file**.

---

# Task B – Candidate Context Ranking

## Objective

Task B focuses on determining how well candidate conversational contexts match the context surrounding a gap in a conversation.

For each conversation gap, candidate turns are evaluated to determine which candidate is more relevant to the missing/context-dependent information.

---

## Approach

Task B uses **BM25**, a classical information retrieval algorithm.

BM25 measures the relevance between a query and a document based on term frequency, inverse document frequency, and document length.

In our implementation, conversational turns are treated as text documents and the surrounding context is treated as the query.

---

## BM25 Pipeline

```text
Conversation
     │
     ▼
Identify conversation gap
     │
     ▼
Extract context before the gap
     │
     ▼
Extract context after the gap
     │
     ▼
Compare candidate context
     │
     ├───────────────┐
     ▼               ▼
BM25 Before      BM25 After
     │               │
     └───────┬───────┘
             ▼
      Combined Scoring
             │
             ▼
       Candidate Ranking
```

---

# BM25 Features

For each candidate, we calculate relevance scores using the surrounding conversation.

The generated dataset contains the following features:

```text
episode_id
gap_id
candidate_id

bm25_before
bm25_after
bm25_combined

bm25_max_before
bm25_max_after

bm25_mean_before
bm25_mean_after
```

These features provide different views of candidate relevance.

---

## BM25 Before

`bm25_before` measures how strongly the candidate matches the conversational context appearing **before the gap**.

This helps identify whether the candidate is related to information that was already discussed.

---

## BM25 After

`bm25_after` measures how strongly the candidate matches the conversational context appearing **after the gap**.

This is useful because later conversation can provide clues about what information was missing or relevant.

---

## BM25 Combined

`bm25_combined` combines the relevance information from the before and after contexts.

This provides a broader measure of candidate relevance.

---

## Maximum and Mean Scores

We also calculate:

```text
bm25_max_before
bm25_max_after
bm25_mean_before
bm25_mean_after
```

These capture different aspects of relevance.

### Maximum Score

The maximum score identifies whether a candidate has a very strong match with at least one conversational turn.

### Mean Score

The mean score measures the candidate's overall relevance across multiple context turns.

---

# Task B Dataset

The generated BM25 feature dataset contains:

```text
Rows: 434

Columns: 10
```

The columns are:

```text
episode_id
gap_id
candidate_id
bm25_before
bm25_after
bm25_combined
bm25_max_before
bm25_max_after
bm25_mean_before
bm25_mean_after
```

Example identifiers follow the format:

```text
C004-G01
```

where:

* `C004` represents the conversation/episode.
* `G01` represents the conversation gap.

---

# Technologies Used

## Python

The complete NLP pipeline is implemented in Python.

Python is used for:

* Data processing
* Text preprocessing
* Embedding generation
* Similarity calculation
* BM25 scoring
* CSV generation
* Evaluation
* Transcript generation

---

## Pandas

Used for:

* Reading CSV datasets
* Data manipulation
* Filtering conversations
* Creating feature datasets
* Writing output CSV files

---

## Sentence Transformers

Used for semantic text representation.

The models convert conversational text into numerical embeddings that can be compared using semantic similarity.

---

## BM25

Used for lexical information retrieval and candidate relevance scoring in Task B.

BM25 is particularly useful when important terms appear in both the candidate context and surrounding conversation.

---

# Overall System

The project combines two different NLP perspectives.

```text
                Context Lag Problem
                       │
              ┌────────┴────────┐
              │                 │
           Task A             Task B
              │                 │
              ▼                 ▼
       Semantic Search       BM25 Retrieval
              │                 │
              ▼                 ▼
   Sentence Transformers      BM25
              │                 │
              ▼                 ▼
      Semantic Similarity    Lexical Relevance
              │                 │
              └────────┬────────┘
                       ▼
              Context Identification
```

---

# Why Two Different Approaches?

The two approaches capture different types of relationships.

### Sentence Transformers

Focus on:

* Semantic meaning
* Paraphrases
* Similar concepts
* Contextual relationships

Example:

```text
"How much did the phone cost?"

"What's the price of the mobile?"
```

These sentences use different wording but have similar meaning.

---

### BM25

Focuses on:

* Keyword overlap
* Important terms
* Term frequency
* Rare/high-information words

Example:

```text
Query:
"Delhi flight Monday"

Candidate:
"Your Delhi flight is scheduled for Monday."
```

BM25 can identify the strong lexical overlap.

---

# Advantages of the Combined Approach

Using both semantic and lexical approaches provides complementary information.

```text
Sentence Transformers
        +
      BM25
        ↓
Semantic + Lexical Evidence
        ↓
Better Context Understanding
```

This is useful because conversational context can contain both:

* Exact references to previously mentioned entities.
* Paraphrased references to earlier information.

---

# Evaluation

The models and approaches were evaluated against the available task evaluation setup.

For Task A, Sentence Transformer approaches were evaluated for their ability to identify the correct contextual relationship.

For Task B, BM25-derived features were generated for the candidate contexts and analyzed for their ability to distinguish relevant candidates.

The final presentation reports the achieved accuracy of the selected models.

---

# Running the Project

## 1. Clone the Repository

```bash
git clone https://github.com/MANOJKUMAR-AS/Dataquest.git
cd Dataquest
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If a requirements file is not available, the main libraries used include:

```bash
pip install pandas
pip install sentence-transformers
pip install rank-bm25
```

---

# Running Task A

Run the Task A processing/prediction scripts provided in the Task A directory.

The generated prediction file is:

```text
outputs/submission_taskA.csv
```

Judge-friendly conversation files are:

```text
outputs/taskA_transcripts.csv
outputs/taskA_transcripts.txt
```

---

# Running Task B

The BM25 implementation can be tested using:

```bash
python test_bm25.py
```

The script processes the conversation candidates and generates BM25-based relevance features.

The resulting feature set contains:

```text
434 rows
10 columns
```

---

# Example Task B Output

```text
episode_id    gap_id    candidate_id    bm25_before    bm25_after
C004          G01       ...             ...            ...
```

Additional combined, maximum, and mean BM25 features are also generated.

---

# Project Workflow

The complete workflow is:

```text
                 Input Conversations
                         │
                         ▼
                 Data Preprocessing
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
           TASK A                 TASK B
              │                     │
              ▼                     ▼
     Sentence Transformers         BM25
              │                     │
              ▼                     ▼
     Semantic Similarity       Relevance Scores
              │                     │
              ▼                     ▼
     Context Prediction       Candidate Features
              │                     │
              └──────────┬──────────┘
                         ▼
                    Evaluation
                         │
                         ▼
                    Final Outputs
```

---

# Key Contributions

### Task A

* Applied Sentence Transformer-based semantic similarity.
* Generated contextual embeddings.
* Identified relevant conversational context.
* Generated the required submission file.
* Created judge-friendly conversation transcripts for inspection.

### Task B

* Implemented BM25-based conversational retrieval.
* Compared candidates against context before and after conversation gaps.
* Generated combined, maximum, and mean BM25 features.
* Created a structured candidate-ranking dataset.
* Tested and inspected BM25 outputs across conversation gaps.

---

# Project Highlights

The project demonstrates the use of both **modern semantic NLP** and **traditional information retrieval** techniques.

### Task A

**Sentence Transformers → Semantic Context Matching**

### Task B

**BM25 → Lexical Context Relevance**

Together, these approaches provide two complementary ways of addressing context lag in conversational systems.

---

# Conclusion

The DataQuest project investigates the **Context Lag Problem** using practical NLP and information-retrieval techniques.

Task A uses **Sentence Transformers** to capture semantic relationships between conversational turns.

Task B uses **BM25** to measure lexical relevance between candidate contexts and the surrounding conversation.

The resulting pipeline provides:

* Semantic context matching
* Lexical relevance scoring
* Candidate ranking features
* Structured predictions
* Judge-friendly outputs

This demonstrates how different NLP techniques can be applied to identify and analyze missing or relevant context in multi-turn conversations.
