-- NLP-English Kingdom Symbol Table
-- Multi-typed GEP symbols for natural language encoding
-- Kingdom ID: 3
-- Each symbol is a verb/relation with typed arity slots (many-hot encoded)
-- and a proposed emoji for the grammar

-- ─────────────────────────────────────────────────────────────────────────────
-- SYMBOLS TABLE (non-terminals / functions)
-- ─────────────────────────────────────────────────────────────────────────────
-- Column guide:
--   emoji        : proposed grammar glyph for the Phylo display
--   alias        : Phylo output verb label
--   dep_patterns : spaCy dep labels that activate this symbol (pipe-separated)
--   in_*         : typed input arity (many-hot) — 0 = type not accepted
--   out_*        : typed output arity — which Phylo clause type this produces
--
-- Output Phylo clause types:
--   out_ENTITY    📦  entity declaration / classification
--   out_RELATION  🔗  relational statement (A relates-to B)
--   out_METRIC    📊  quantitative measurement
--   out_EVENT     ⚡  event / occurrence
--   out_PROCEDURE ⚙️  process / method
--   out_NARRATIVE 📜  untyped / fallback
-- ─────────────────────────────────────────────────────────────────────────────

INSERT INTO symbols (
    kingdom, symbol, symbol_name, alias, emoji,

    -- Input arities: NLP types (many-hot)
    in_PERSON, in_ORG, in_GPE,   in_LOC, in_NORP, in_FAC,
    in_EVENT,  in_PRODUCT, in_DATE, in_TIME, in_MONEY, in_QUANTITY,
    in_CARDINAL, in_PERCENT, in_NP,  in_AP,  in_CLAUSE, in_VERB,

    -- Output arities: Phylo clause types
    out_ENTITY, out_RELATION, out_METRIC, out_EVENT, out_PROCEDURE, out_NARRATIVE
)
VALUES

-- ─── FALLBACK ────────────────────────────────────────────────────────────────
--  ID  name        alias     emoji
(  'NLP-English',  0, 'narrative',   'SAYS',       '📜',
-- P   O   G   L   N   F   Ev  Pr  D   Ti  Mo  Q   Ca  Pc  NP  AP  Cl  Vb
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
-- ENT  REL  MET  EVT  PRC  NAR
   0,   0,   0,   0,   0,   1 ),

-- ─── ACQUISITION / COMMERCE ──────────────────────────────────────────────────
(  'NLP-English',  1, 'acquire',     'ACQUIRED',   '💰',
   0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,
   0,  1,  0,  0,  0,  0 ),

(  'NLP-English', 14, 'fork',        'FORKED',     '🍴',
   0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  0,  0,  0,  0 ),

(  'NLP-English', 16, 'agree',       'AGREED',     '🤝',
   0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  1,  0,
   0,  1,  0,  0,  0,  0 ),

-- ─── MOVEMENT / PRESENCE ─────────────────────────────────────────────────────
(  'NLP-English',  2, 'visit',       'VISITED',    '✈️',
   1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  1,  0,  0 ),

(  'NLP-English',  7, 'meet',        'MET',        '👥',
   2,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  1,  0,  0 ),

-- ─── CLASSIFICATION / IDENTITY ───────────────────────────────────────────────
(  'NLP-English',  3, 'is-a',        'IS',         '🏷️',
   0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,
   1,  0,  0,  0,  0,  0 ),

(  'NLP-English', 13, 'appoint',     'APPOINTED',  '👔',
   1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,
   0,  1,  0,  0,  0,  0 ),

-- ─── MEASUREMENT / METRICS ───────────────────────────────────────────────────
(  'NLP-English',  4, 'measure',     'MEASURES',   '📏',
   0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0,
   0,  0,  1,  0,  0,  0 ),

-- ─── COMPETITION / CONFLICT ──────────────────────────────────────────────────
(  'NLP-English',  5, 'compete',     'COMPETING',  '🏆',
   0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,
   0,  1,  0,  0,  0,  0 ),

(  'NLP-English', 10, 'strike',      'STRUCK',     '💥',
   0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  1,  0,  0 ),

(  'NLP-English', 18, 'divide',      'DIVIDED',    '⚔️',
   0,  1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,
   0,  1,  0,  0,  0,  0 ),

-- ─── PERFORMANCE / EVENTS ────────────────────────────────────────────────────
(  'NLP-English',  6, 'perform-at',  'PERFORMED',  '🎭',
   1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  1,  0,  0 ),

(  'NLP-English', 17, 'die',         'DIED',       '💀',
   1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,
   0,  0,  0,  1,  0,  0 ),

-- ─── COMMUNICATION ───────────────────────────────────────────────────────────
(  'NLP-English', 11, 'urge',        'URGED',      '📢',
   0,  1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,
   0,  1,  0,  0,  0,  0 ),

(  'NLP-English', 12, 'announce',    'ANNOUNCED',  '📣',
   0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,
   0,  1,  0,  0,  0,  0 ),

(  'NLP-English', 15, 'praise',      'PRAISED',    '👏',
   1,  1,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,
   0,  1,  0,  1,  0,  0 ),

-- ─── PRODUCTION / CREATION ───────────────────────────────────────────────────
(  'NLP-English',  9, 'create',      'CREATED',    '🔨',
   0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,
   0,  1,  0,  0,  0,  0 ),

-- ─── SUSPENSION / PAUSE ──────────────────────────────────────────────────────
(  'NLP-English',  8, 'pause',       'PAUSED',     '⏸️',
   0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,
   0,  1,  0,  0,  0,  0 ),

-- ─── UNKNOWN VERB (staging / identity) ───────────────────────────────────────
-- IsVerb captures any unrecognised ROOT verb.
-- The verb lemma itself occupies in_VERB=1 (first tail position).
-- Subsequent positions are open-typed NLP args from the dep parse.
-- When enough instances accumulate, IsVerb("X") promotes to a named symbol.
(  'NLP-English', 99, 'isverb',      'IsVerb',     '🏃',
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
   0,  1,  0,  0,  0,  0 );


-- ─────────────────────────────────────────────────────────────────────────────
-- TERMINALS TABLE (typed leaf nodes)
-- ─────────────────────────────────────────────────────────────────────────────
-- Terminals have NEGATIVE symbol IDs.
-- Static terminals are pre-loaded constants.
-- Dynamic terminals are inserted at encode time from spaCy NER output.
-- ─────────────────────────────────────────────────────────────────────────────

INSERT INTO terminals (
    kingdom, symbol, symbol_name, alias, value,
    out_PERSON, out_ORG, out_GPE, out_LOC, out_NORP, out_FAC,
    out_EVENT,  out_PRODUCT, out_DATE, out_TIME, out_MONEY, out_QUANTITY,
    out_CARDINAL, out_PERCENT, out_NP, out_AP, out_CLAUSE, out_VERB
)
VALUES

-- ─── STATIC WELL-KNOWN ENTITIES ──────────────────────────────────────────────
-- (populated by the NER pre-load process from a curated gazetteer)
-- Ephemeral NER-derived terminals are inserted dynamically at encode time
-- with auto-assigned negative IDs below -1000.

-- ─── NUMERIC CONSTANTS ───────────────────────────────────────────────────────
( 'NLP-English', -1,  'Zero',    '0',  '0',   0,0,0,0,0,0, 0,0,0,0,0,0, 1,0,0,0,0,0 ),
( 'NLP-English', -2,  'One',     '1',  '1',   0,0,0,0,0,0, 0,0,0,0,0,0, 1,0,0,0,0,0 ),
( 'NLP-English', -3,  'Two',     '2',  '2',   0,0,0,0,0,0, 0,0,0,0,0,0, 1,0,0,0,0,0 ),
( 'NLP-English', -4,  'Three',   '3',  '3',   0,0,0,0,0,0, 0,0,0,0,0,0, 1,0,0,0,0,0 ),
( 'NLP-English', -5,  'Four',    '4',  '4',   0,0,0,0,0,0, 0,0,0,0,0,0, 1,0,0,0,0,0 ),
( 'NLP-English', -6,  'Five',    '5',  '5',   0,0,0,0,0,0, 0,0,0,0,0,0, 1,0,0,0,0,0 ),
( 'NLP-English', -7,  'Six',     '6',  '6',   0,0,0,0,0,0, 0,0,0,0,0,0, 1,0,0,0,0,0 ),
( 'NLP-English', -8,  'Seven',   '7',  '7',   0,0,0,0,0,0, 0,0,0,0,0,0, 1,0,0,0,0,0 ),
( 'NLP-English', -9,  'Eight',   '8',  '8',   0,0,0,0,0,0, 0,0,0,0,0,0, 1,0,0,0,0,0 ),
( 'NLP-English', -10, 'Nine',    '9',  '9',   0,0,0,0,0,0, 0,0,0,0,0,0, 1,0,0,0,0,0 ),

-- ─── UNKNOWN TERMINALS (placeholder for NER-assigned types) ──────────────────
-- These are not inserted statically; they are generated at encode time.
-- Negative IDs < -1000 are reserved for ephemeral NER terminals.
-- Schema:
--   symbol      = -(1000 + sequence)
--   symbol_name = raw text span ("Barack Obama", "UK startup", "$1 billion")
--   out_*       = 1 for the NER label type, 0 for all others
--   value       = raw text span

-- ─── NULL / PADDING terminal ─────────────────────────────────────────────────
-- Used to pad the tail to fixed chromosome length when fewer typed terminals
-- are available than TailLength requires.
( 'NLP-English', -999, 'NULL', 'NULL', '',  0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,1,0,0,0 );


-- ─────────────────────────────────────────────────────────────────────────────
-- EMOJI GRAMMAR REFERENCE
-- ─────────────────────────────────────────────────────────────────────────────
--
-- Functions (symbols):
--   🏃  IsVerb       action in progress — staging area for new symbols
--   💰  acquire      commercial acquisition / purchase
--   🍴  fork         split / spin-off / partition
--   🤝  agree        agreement / deal / treaty
--   ✈️  visit        physical presence at a location
--   👥  meet         face-to-face encounter
--   🏷️  is-a         classification / identity declaration
--   👔  appoint      role assignment / appointment
--   📏  measure      quantitative measurement / metric
--   🏆  compete      competition / rivalry
--   💥  strike       attack / hit / impact
--   ⚔️  divide       split / polarise / separate
--   🎭  perform-at   performance / appearance at event
--   💀  die          death / cessation
--   📢  urge         pressure / call-to-action
--   📣  announce     public declaration
--   👏  praise       positive evaluation
--   🔨  create       creation / production
--   ⏸️  pause        suspension / halt
--   📜  narrative    fallback / unclassified statement
--
-- Output clause types (Phylo):
--   📦  ENTITY       X is-a Y / X has property P
--   🔗  RELATION     X VERB Y
--   📊  METRIC       X = value unit
--   ⚡  EVENT        X VERB Y :at TIME/PLACE
--   ⚙️  PROCEDURE    instruction / process
--   📜  NARRATIVE    general statement (fallback)
--
-- Input terminal types (NER):
--   👤  PERSON       named individual
--   🏢  ORG          organisation
--   🌍  GPE          geopolitical entity
--   🗺️  LOC          geographic location
--   🎌  NORP         nationality / religion / political group
--   🏗️  FAC          facility / building
--   🎪  EVENT        named event
--   📱  PRODUCT      product / artifact
--   📅  DATE         date expression
--   🕐  TIME         time expression
--   💵  MONEY        monetary value
--   ⚖️  QUANTITY     physical quantity with unit
--   🔢  CARDINAL     bare numeral
--   💯  PERCENT      percentage
--   🔤  NP           generic noun phrase
--   🔡  AP           adjectival phrase
--   📎  CLAUSE       subordinate clause
--   🔠  VERB         verb lemma (IsVerb staging)
-- ─────────────────────────────────────────────────────────────────────────────
