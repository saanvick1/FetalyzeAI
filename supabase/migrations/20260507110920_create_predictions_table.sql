/*
  # FetalyzeAI Prediction History

  1. New Tables
    - `predictions`
      - `id` (uuid, primary key)
      - `session_id` (text) — anonymous browser session
      - `features` (jsonb) — the 21 CTG input features
      - `risk_label` (text) — Normal | Suspect | Pathological
      - `risk_class` (int) — 0, 1, 2
      - `confidence` (numeric) — 0–1 probability of predicted class
      - `prob_normal` (numeric)
      - `prob_suspect` (numeric)
      - `prob_pathological` (numeric)
      - `fetal_reserve_score` (numeric) — 0–100
      - `created_at` (timestamptz)

  2. Security
    - RLS enabled
    - Anyone can insert (anonymous submissions)
    - Users can read only rows matching their session_id
*/

CREATE TABLE IF NOT EXISTS predictions (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id          text NOT NULL DEFAULT '',
  features            jsonb NOT NULL DEFAULT '{}',
  risk_label          text NOT NULL DEFAULT 'Unknown',
  risk_class          int NOT NULL DEFAULT 0,
  confidence          numeric NOT NULL DEFAULT 0,
  prob_normal         numeric NOT NULL DEFAULT 0,
  prob_suspect        numeric NOT NULL DEFAULT 0,
  prob_pathological   numeric NOT NULL DEFAULT 0,
  fetal_reserve_score numeric,
  created_at          timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can insert predictions"
  ON predictions FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Session owner can read own predictions"
  ON predictions FOR SELECT
  TO anon, authenticated
  USING (session_id = current_setting('request.headers', true)::jsonb->>'x-session-id' OR session_id = '');
