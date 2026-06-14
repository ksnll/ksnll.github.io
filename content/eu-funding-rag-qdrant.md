+++
title = "Building a RAG system for EU funding discovery"
date = 2026-06-14
description = "A practical look at building a RAG system for EU funding discovery with Qdrant, structured metadata, reranking and grounded answers."
+++

EU funding data is public, useful, and still surprisingly hard to search.

The information is spread across different sources: the official EU Funding & Tenders Portal, programme pages, PDFs, national contact points, and other datasets such as KEEP.eu.
Each source is useful, but none of them maps directly to the way a normal person asks a question.

A user will not usually ask:

```text
Show me Horizon Europe Cluster 6 calls with destination FARM2FORK and eligible action type RIA.
```

They will ask something more like:

```text
We are a Cyprus-based SME building an AI tool that helps farmers detect crop disease from satellite and drone images. Is there any EU funding for this?
```

This is the kind of problem where a RAG system can be useful.

Not because the model knows EU funding. We actually do not want that. We want the model to search the right source material, reason over it, and produce an answer grounded in the data we indexed.

<!-- more -->

This also helps with one of the main problems with LLMs: hallucinations. If the model answers from its own training data, it can invent deadlines, eligibility conditions, budgets, or programmes that are no longer active. If we make retrieval, metadata and citations part of the system design, we can reduce that risk quite a lot.

In this article, we'll look at the shape of a practical RAG system for EU funding discovery using Qdrant as the vector database.

The goal is not to build a demo that works on one nice query. The goal is to design something that survives the messy parts:

- funding opportunities come from multiple datasets
- calls are long and full of formal language
- eligibility is often hidden in PDFs
- deadlines and budgets matter
- users ask vague questions
- similar text is not always the same as relevant opportunity
- unsupported answers are dangerous

Let's start with the simplest version and then improve it step by step.

### The naive version

The first version of a RAG system is usually something like this:

1. Split all documents into chunks
2. Embed each chunk
3. Store the embeddings in a vector database
4. Embed the user question
5. Retrieve the closest chunks
6. Send them to an LLM
7. Ask the LLM to answer

In code, it could look like this:

```ts
const query = "Are there EU grants for AI tools in agriculture?";

const queryEmbedding = await embed(query);

const results = await qdrant.search("funding_chunks", {
    vector: queryEmbedding,
    limit: 10,
});

const answer = await claude.messages.create({
    model: "claude-sonnet-4-5",
    max_tokens: 1200,
    messages: [
        {
            role: "user",
            content: `
Answer the question using these documents:

${results.map((r) => r.payload.text).join("\n\n")}

Question: ${query}
            `,
        },
    ],
});
```

This works surprisingly well for a demo.

It also breaks quite quickly.

The first problem is that EU funding discovery is not just a semantic similarity problem.
If someone asks for funding for an AI agriculture project, we need to consider at least:

- topic fit
- applicant type
- geography
- call status
- deadline
- funding instrument
- consortium requirements
- budget
- TRL level
- whether the call funds research, deployment, training, infrastructure, or innovation

A chunk that says `artificial intelligence in agriculture` may be semantically close, but useless if the call closed two years ago or if only public authorities can apply.

So we need to model the data more explicitly.

### Store chunks, but do not lose the call

A common mistake is to treat chunks as the only unit of retrieval.

For EU funding this is not enough. We need chunks for semantic search, but we also need the parent opportunity as a first-class object.

For example, we might have a schema like this:

```ts
type FundingOpportunity = {
    id: string;
    title: string;
    source: "funding_tenders" | "keep_eu" | "programme_page" | "manual";
    programme?: string;
    topicCode?: string;
    status?: "open" | "upcoming" | "closed";
    deadline?: string;
    budget?: string;
    countries?: string[];
    applicantTypes?: string[];
    url: string;
};

type FundingChunk = {
    id: string;
    opportunityId: string;
    text: string;
    section:
        | "summary"
        | "scope"
        | "eligibility"
        | "expected_outcome"
        | "budget"
        | "deadline"
        | "partners"
        | "other";
    chunkIndex: number;
};
```

This is just an example schema. The real shape depends on your sources and how much metadata you can extract reliably.

For instance, the official Funding & Tenders Portal gives you structured fields for some things, while PDFs and programme pages may need extraction. KEEP.eu is useful for understanding past projects and partnerships, but it is a different type of dataset than an open call.

The important part is to keep both levels:

- the opportunity or project as the parent object
- the text chunks as the searchable evidence

In Qdrant, each point stores the vector for a chunk, but the payload keeps enough metadata to filter and explain the result:

```ts
await qdrant.upsert("funding_chunks", {
    points: [
        {
            id: chunk.id,
            vector: embedding,
            payload: {
                opportunity_id: opportunity.id,
                title: opportunity.title,
                source: opportunity.source,
                programme: opportunity.programme,
                topic_code: opportunity.topicCode,
                status: opportunity.status,
                deadline: opportunity.deadline,
                applicant_types: opportunity.applicantTypes,
                countries: opportunity.countries,
                section: chunk.section,
                url: opportunity.url,
                text: chunk.text,
            },
        },
    ],
});
```

Now we can search semantically, but still apply structured constraints.

For example, if the user asks for open calls, we should not rely on the model to ignore closed opportunities. We can filter them before they reach the model:

```ts
const results = await qdrant.search("funding_chunks", {
    vector: queryEmbedding,
    limit: 30,
    filter: {
        must: [
            {
                key: "status",
                match: { value: "open" },
            },
        ],
    },
});
```

This is already better.

But we still have another problem: the user may not use the same vocabulary as the funding portal.

### Search queries are not user questions

Users describe their project in plain language:

```text
I have a startup doing satellite analytics for farms. Is there money for this?
```

A call might use very different terms:

```text
Earth observation, Copernicus downstream services, precision farming,
agri-food data spaces, digital technologies for sustainable agriculture.
```

The user did not say `Copernicus`, `Earth observation` or `data spaces`.
A vector search may still find the right thing, but it is not something I would rely on completely.

A useful step is to rewrite the user question into a retrieval query.

Not an answer. Just a better search query.

```ts
const rewritePrompt = `
You are helping search a database of EU funding opportunities.

Rewrite the user's question into a concise search query that includes:
- relevant technical terms
- likely EU policy or programme vocabulary
- synonyms
- applicant type if mentioned
- geography if mentioned

Do not answer the question.

<user_question>
${userQuestion}
</user_question>
`;
```

For the startup example, a rewritten query might be:

```text
SME startup satellite analytics agriculture precision farming Earth observation Copernicus agri-food digital technologies sustainable agriculture EU funding Cyprus
```

In practice, I like to keep both:

- the original user question, for the final answer
- the rewritten query, for retrieval

This makes the search less dependent on the user's vocabulary.

### Retrieve more than you need

Another common mistake is retrieving exactly the number of chunks that you want to send to the model.

For example:

```ts
limit: 5
```

This is usually too small.

The first retrieval step should be recall-oriented. We want to collect plausible candidates. Then we can deduplicate, group by opportunity, rerank, and only then select the final context.

For example:

```ts
const candidates = await qdrant.search("funding_chunks", {
    vector: searchEmbedding,
    limit: 50,
    filter,
});
```

Then we group by `opportunity_id`:

```ts
const byOpportunity = new Map<string, FundingChunkResult[]>();

for (const result of candidates) {
    const opportunityId = result.payload.opportunity_id;
    if (!byOpportunity.has(opportunityId)) {
        byOpportunity.set(opportunityId, []);
    }
    byOpportunity.get(opportunityId)!.push(result);
}
```

This matters because if one opportunity has ten similar chunks, it can crowd out every other opportunity.

I do not want the retrieval step to return ten chunks from the same call. I want it to return a good set of candidate opportunities.

A simple scoring approach could be:

```ts
const opportunityCandidates = [...byOpportunity.entries()].map(
    ([opportunityId, chunks]) => {
        const bestScore = Math.max(...chunks.map((c) => c.score));
        const sections = new Set(chunks.map((c) => c.payload.section));

        return {
            opportunityId,
            bestScore,
            sectionCoverage: sections.size,
            chunks: chunks.slice(0, 4),
        };
    },
);

opportunityCandidates.sort((a, b) => b.bestScore - a.bestScore);
```

This is not perfect, but it is already better than passing raw chunks to the model.

### Rerank for actual usefulness

Semantic similarity is not the same as usefulness.

For EU funding, a good result should help answer:

- is the opportunity relevant to the project?
- is the applicant type plausible?
- is it open or upcoming?
- what should the user check next?
- is this an actual call, a past project, or supporting background information?

This is where a reranking step helps.

You can use a dedicated reranker, but you can also start with an LLM-based reranker. The important part is to ask for a structured judgment and to keep it inspectable.

```ts
const rerankPrompt = `
You are ranking EU funding opportunities for relevance.

User question:
<question>
${userQuestion}
</question>

Candidate opportunities:
<candidates>
${opportunityCandidates
    .map(
        (candidate, index) => `
<candidate index="${index}">
<title>${candidate.title}</title>
<source>${candidate.source}</source>
<programme>${candidate.programme}</programme>
<status>${candidate.status ?? "unknown"}</status>
<deadline>${candidate.deadline ?? "unknown"}</deadline>
<snippets>
${candidate.chunks
    .map(
        (chunk) =>
            `<snippet section="${chunk.payload.section}">${chunk.payload.text}</snippet>`,
    )
    .join("\n")}
</snippets>
</candidate>
`,
    )
    .join("\n")}
</candidates>

For each candidate, score it from 0 to 5:
- 5: directly relevant and likely actionable
- 4: relevant but needs eligibility checks
- 3: adjacent, possibly useful
- 2: weak relevance
- 1: mostly irrelevant
- 0: not relevant

Return JSON only.
`;
```

I like this pattern because it makes failures easier to debug.

If the answer is bad, we can check:

- did the vector search retrieve the right things?
- did grouping lose useful chunks?
- did reranking choose the wrong opportunities?
- did the final model ignore the evidence?

Without these intermediate steps, RAG systems can become very difficult to reason about.

### Build the context explicitly

When sending the final context to the model, I prefer not to dump raw chunks.

The model should not have to guess what is a title, what is a deadline and what is a source URL.

A structured context works much better:

```xml
<opportunity id="HORIZON-CL6-2025-01">
  <title>Digital tools for sustainable agriculture</title>
  <source>funding_tenders</source>
  <programme>Horizon Europe</programme>
  <status>open</status>
  <deadline>2025-09-18</deadline>
  <url>{opportunity.url}</url>

  <snippet section="scope">
    Projects should develop digital technologies for sustainable agriculture...
  </snippet>

  <snippet section="eligibility">
    Proposals must be submitted by a consortium of at least three legal entities...
  </snippet>

  <snippet section="expected_outcome">
    Proposals are expected to improve productivity and environmental performance...
  </snippet>
</opportunity>
```

Clear delimiters and explicit source structure are a simple way to improve the final answer.
They also reduce hallucinations, because the model is less tempted to fill in missing fields from memory when the available fields are clearly defined.

### The final answer prompt

The final prompt should be strict.

We are not asking the model to be creative. We are asking it to answer from retrieved sources.

```ts
const answerPrompt = `
You are helping users discover EU funding opportunities.

Answer the user's question using only the information inside <sources>.
If the sources are not enough, say what is missing.
Do not invent eligibility, deadlines, budgets, funding rates, or requirements.
Cite the relevant source URL for each recommendation.

<user_question>
${userQuestion}
</user_question>

<sources>
${contextXml}
</sources>

Answer with this structure:

1. Short answer
2. Best matches
3. Why they match
4. Eligibility and deadline checks
5. Recommended next steps
`;
```

This is one of the most important parts of the system.

If a deadline is not in the sources, the answer should not include a guessed deadline.
If the eligibility section is missing, the answer should say that eligibility needs to be checked.

This is less flashy than a confident answer, but it is much more useful.

### A realistic example

Let's imagine the user asks:

```text
We are a Cyprus-based SME building an AI system that detects crop disease from satellite and drone images. We have a prototype and want to know if there are EU funding opportunities in the next 6 months.
```

The rewritten query might be:

```text
Cyprus SME AI crop disease detection satellite drone imagery precision agriculture Earth observation Copernicus digital farming innovation funding next 6 months
```

After searching Qdrant, grouping by opportunity, and reranking, the debug output could look like this:

```json
{
  "query_type": "discover_opportunities",
  "rewritten_query": "Cyprus SME AI crop disease detection satellite drone imagery precision agriculture Earth observation Copernicus digital farming innovation funding next 6 months",
  "filters": {
    "status": ["open", "upcoming"]
  },
  "retrieved_chunks": 50,
  "candidate_opportunities": 12,
  "reranked": [
    {
      "title": "Digital and data technologies for sustainable agriculture",
      "source": "funding_tenders",
      "score": 5,
      "reason": "Mentions digital technologies, data-driven farming and sustainable agriculture. Eligibility and consortium rules must be checked."
    },
    {
      "title": "Copernicus downstream applications for agri-food and environmental monitoring",
      "source": "funding_tenders",
      "score": 4,
      "reason": "Strong fit for satellite imagery and Earth observation, but the call may require specific consortium composition."
    },
    {
      "title": "Past KEEP.eu projects on precision farming and remote sensing",
      "source": "keep_eu",
      "score": 3,
      "reason": "Not a funding call, but useful to discover similar funded projects and possible partners."
    }
  ]
}
```

A good final answer could be:

```text
Short answer

Yes, there are a few promising directions, but the best fit depends on whether you want to fund R&D, a pilot deployment, or partner discovery.

Best matches

1. Digital and data technologies for sustainable agriculture

This looks like the strongest match because the scope mentions digital farming, data-driven agricultural tools and sustainability outcomes.

Why it matches:
- your project uses AI in agriculture
- the use case is related to crop monitoring and precision farming
- the opportunity is about digital technologies in the agri-food domain

Checks:
- verify whether a Cyprus SME can participate directly
- verify whether a consortium is required
- check the expected TRL range
- deadline: 2025-09-18

Source: Funding & Tenders Portal, call page URL from the retrieved source

2. Copernicus downstream applications for agri-food and environmental monitoring

This is relevant because the project uses satellite imagery and could fit an Earth observation or Copernicus downstream services angle.

Checks:
- confirm whether drone imagery is within scope or only satellite data is accepted
- verify consortium requirements
- check whether the call is for R&D or market deployment

Source: Funding & Tenders Portal, call page URL from the retrieved source

3. Similar KEEP.eu projects on precision farming and remote sensing

These are not open calls, but they are useful for understanding what has been funded before and which organisations have worked in this space.

Recommended next steps

- shortlist calls where SMEs are explicitly eligible
- check consortium requirements before spending time on a proposal
- use KEEP.eu results to identify previous projects and potential partners
- compare deadlines against the next 6 months
```

The important thing is that the answer is useful but cautious.

It does not say `you are eligible` unless the source says that.
It does not invent a funding rate.
It separates active funding opportunities from historical project data.
That distinction matters a lot when mixing datasets like Funding & Tenders and KEEP.eu.

### Filters are part of the product

Metadata filters are not just an implementation detail. They are part of the user experience.

A user may ask:

```text
Show me open calls for SMEs in Cyprus.
```

This should affect retrieval:

```ts
const filter = {
    must: [
        { key: "status", match: { value: "open" } },
    ],
    should: [
        { key: "applicant_types", match: { any: ["SME"] } },
        { key: "countries", match: { any: ["Cyprus", "EU Member States"] } },
    ],
};
```

But we need to be careful.

I am comfortable hard-filtering closed calls when the user asks for open opportunities.
I am more careful hard-filtering by country or applicant type, because EU eligibility is often expressed indirectly.

For example, a call might say:

```text
Legal entities established in EU Member States and associated countries are eligible.
```

If our metadata extraction did not normalize this correctly to `Cyprus`, a hard country filter could remove a relevant call.

So a practical approach is:

- hard filter on things we trust, like `status`
- soft boost applicant type and geography
- make the final answer explain what needs to be checked

### Chunking matters more than expected

Naive chunking can produce strange results.

If we split every 500 tokens, we might separate:

- the title from the scope
- the deadline from the eligibility section
- the expected outcome from the budget
- the consortium requirements from the type of action

For funding data, it is better to chunk by document structure when possible.

```ts
const sections = [
    "Objective",
    "Scope",
    "Expected Outcome",
    "Eligibility",
    "Budget",
    "Deadline",
    "Type of Action",
];

for (const section of extractedSections) {
    const chunks = splitSection(section.text, {
        maxTokens: 700,
        overlapTokens: 120,
    });

    for (const chunk of chunks) {
        await storeChunk({
            opportunityId,
            section: section.name,
            text: chunk.text,
        });
    }
}
```

The section name becomes important metadata.

If the user asks:

```text
Can my SME apply?
```

then chunks from `eligibility` should be preferred.

If the user asks:

```text
Is this about agriculture?
```

then `scope` and `expected_outcome` are probably more useful.

### Use different retrieval strategies for different questions

Not all questions need the same retrieval path.

For example:

- `Find calls for my project` needs broad retrieval and reranking
- `Is this specific call relevant to me?` needs call-specific retrieval
- `What is the deadline?` should prefer structured metadata
- `Can I apply alone?` should retrieve eligibility sections first
- `Who could we partner with?` may need KEEP.eu projects more than open calls

A small classifier can help:

```ts
type QueryType =
    | "discover_opportunities"
    | "compare_opportunities"
    | "specific_opportunity_question"
    | "deadline_or_budget"
    | "eligibility_check"
    | "partner_discovery";
```

Then we can choose a retrieval strategy.

This feels like overengineering until you try to answer eligibility questions with the same retrieval flow you use for broad discovery.

### Keep citations attached to claims

If the system recommends a call, it should include the source URL.

Even better, claims should be traceable to a section.

Bad:

```text
This call is relevant for AI agriculture projects.
```

Better:

```text
This call is relevant because the scope mentions digital farming and data-driven agricultural tools.
Source: Funding & Tenders Portal, Scope section, {source_url}
```

The payload should make that possible:

```ts
payload: {
    opportunity_id: opportunity.id,
    source: opportunity.source,
    source_url: opportunity.url,
    source_title: opportunity.title,
    section: "scope",
    text: chunk.text,
}
```

This does not eliminate hallucinations entirely, but it gives the system a much better chance of staying grounded.
It also gives users a way to verify the answer.

### Evaluation should start simple

RAG systems are hard to judge by looking at one good answer.

I like to keep a small set of test questions:

```ts
const evalQuestions = [
    {
        question:
            "We are an SME in Cyprus building AI tools for agriculture. What calls should we look at?",
        expectedTopics: ["agriculture", "AI", "SME"],
    },
    {
        question: "Are there open calls for digital education platforms?",
        expectedTopics: ["education", "digital", "open"],
    },
    {
        question: "Can a startup apply alone or do we need a consortium?",
        expectedSections: ["eligibility"],
    },
    {
        question: "Who has received funding before for remote sensing in agriculture?",
        expectedSources: ["keep_eu"],
    },
];
```

For each question, I want to inspect:

- did retrieval find the right opportunities?
- did reranking choose the right ones?
- did the final answer cite sources?
- did the model avoid unsupported claims?
- did it say what is uncertain?

At first, manual evaluation is enough. Later, this can become more systematic.

A useful debug output is:

```json
{
  "question": "We are an SME in Cyprus building AI tools for agriculture. What calls should we look at?",
  "rewritten_query": "SME Cyprus artificial intelligence agriculture precision farming EU funding",
  "filters": {
    "status": ["open", "upcoming"]
  },
  "top_results": [
    {
      "title": "Digital and data technologies for sustainable agriculture",
      "score": 5,
      "sections": ["scope", "eligibility", "expected_outcome"],
      "source": "funding_tenders"
    }
  ]
}
```

This makes the system easier to improve.

### The full flow

Putting it all together, the practical flow looks like this:

```text
User question
   |
   v
Classify question type
   |
   v
Rewrite question for search
   |
   v
Build metadata filters
   |
   v
Search Qdrant for candidate chunks
   |
   v
Group chunks by opportunity
   |
   v
Rerank candidate opportunities
   |
   v
Build structured source context
   |
   v
Generate answer with citations
   |
   v
Return answer + source links + uncertainty
```

In code, roughly:

```ts
async function answerFundingQuestion(userQuestion: string) {
    const queryType = await classifyQuery(userQuestion);

    const rewrittenQuery = await rewriteForSearch(userQuestion);

    const filter = buildFilter(userQuestion, queryType);

    const candidates = await retrieveCandidates({
        query: rewrittenQuery,
        filter,
        limit: 50,
    });

    const grouped = groupByOpportunity(candidates);

    const ranked = await rerankOpportunities({
        userQuestion,
        candidates: grouped,
    });

    const context = buildContextXml(ranked.slice(0, 5));

    const answer = await generateAnswer({
        userQuestion,
        context,
    });

    return {
        answer,
        sources: ranked.map((r) => r.url),
        debug: {
            queryType,
            rewrittenQuery,
            retrieved: candidates.length,
            ranked: ranked.length,
        },
    };
}
```

### What I would not do

There are a few tempting shortcuts I would avoid.

### I would not let the model answer from memory

For funding discovery, this is too risky.

Deadlines, eligibility rules and programme details change. The model should answer from retrieved sources or say that the sources are not enough.

### I would not use vector search alone

Vector search is useful, but funding discovery also needs metadata, filtering, grouping, reranking and citations.

### I would not hide uncertainty

A good answer should say things like:

```text
This appears relevant, but you should verify consortium requirements and national eligibility rules.
```

That is not a weakness. That is the correct behavior.

### I would not treat all datasets as the same thing

The official Funding & Tenders Portal and KEEP.eu answer different questions.

Funding & Tenders is useful for open and upcoming opportunities.
KEEP.eu is useful for historical projects, programme patterns and partner discovery.

Mixing them without labeling the source can produce very confusing answers.

### Outro

RAG is often described as `chat with your documents`, but for EU funding discovery that is not quite enough.

The useful product is closer to a decision-support system:

- retrieve relevant opportunities
- explain why they match
- separate active calls from historical project data
- highlight missing eligibility checks
- cite source material
- avoid pretending to know what is not in the data

Qdrant is a good fit for the retrieval layer because it lets us combine embeddings with metadata filtering. But the quality of the system comes from the parts around it: document structure, query rewriting, grouping by opportunity, reranking, careful prompting and evaluation.

The final goal is not to replace the expert. It is to reduce the amount of repetitive research needed before an expert can make a good decision.

That is where RAG works best: not as magic, but as a practical interface over messy knowledge.
