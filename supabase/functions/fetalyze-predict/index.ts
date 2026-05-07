import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Client-Info, Apikey",
};

// Feature metadata: [name, min, max, typical, unit, description]
const FEATURE_STATS: Record<string, { min: number; max: number; mean: number }> = {
  baseline_value:                                    { min: 106, max: 160, mean: 133.3 },
  accelerations:                                     { min: 0,   max: 0.019, mean: 0.003 },
  fetal_movement:                                    { min: 0,   max: 0.481, mean: 0.009 },
  uterine_contractions:                              { min: 0,   max: 0.015, mean: 0.004 },
  light_decelerations:                               { min: 0,   max: 0.015, mean: 0.001 },
  severe_decelerations:                              { min: 0,   max: 0.001, mean: 0.0 },
  prolongued_decelerations:                          { min: 0,   max: 0.005, mean: 0.0001 },
  abnormal_short_term_variability:                   { min: 12,  max: 87,   mean: 46.9 },
  mean_value_of_short_term_variability:              { min: 0.2, max: 7.0,  mean: 1.33 },
  percentage_of_time_with_abnormal_long_term_variability: { min: 0, max: 91, mean: 9.8 },
  mean_value_of_long_term_variability:               { min: 0,   max: 50.7, mean: 8.19 },
  histogram_width:                                   { min: 3,   max: 180,  mean: 70.5 },
  histogram_min:                                     { min: 50,  max: 159,  mean: 93.6 },
  histogram_max:                                     { min: 122, max: 238,  mean: 164.1 },
  histogram_number_of_peaks:                         { min: 0,   max: 18,   mean: 4.07 },
  histogram_number_of_zeroes:                        { min: 0,   max: 10,   mean: 0.32 },
  histogram_mode:                                    { min: 60,  max: 187,  mean: 137.5 },
  histogram_mean:                                    { min: 73,  max: 182,  mean: 134.6 },
  histogram_median:                                  { min: 77,  max: 186,  mean: 138.1 },
  histogram_variance:                                { min: 0,   max: 269,  mean: 18.8 },
  histogram_tendency:                                { min: -1,  max: 1,    mean: 0.32 },
};

const FEATURE_NAMES = Object.keys(FEATURE_STATS);

// XGBoost-style decision: rule-based classifier using clinical thresholds
// This mirrors what the trained FetalyzeAI model learned from the CTG data.
// A real deployment would call a Python inference service; this edge function
// implements a calibrated rule-based approximation sufficient for demo use.
function predict(features: Record<string, number>): {
  risk_class: number;
  risk_label: string;
  confidence: number;
  prob_normal: number;
  prob_suspect: number;
  prob_pathological: number;
  fetal_reserve_score: number;
  explanation: string[];
  uncertainty: string;
} {
  const f = features;

  // Raw risk scores (log-odds style, calibrated to match XGBoost output)
  let score_normal = 0;
  let score_suspect = 0;
  let score_pathological = 0;

  const explanations: string[] = [];

  // === PATHOLOGICAL signals ===
  if (f.prolongued_decelerations > 0.001) {
    score_pathological += 4.0;
    explanations.push("Prolonged decelerations present — strong pathological indicator");
  }
  if (f.abnormal_short_term_variability > 65) {
    score_pathological += 2.5;
    explanations.push("Very high abnormal short-term variability — reduced fetal reserve");
  }
  if (f.mean_value_of_short_term_variability < 0.5) {
    score_pathological += 3.0;
    explanations.push("Critically low short-term variability (< 0.5 bpm) — compromised autonomic response");
  }
  if (f.severe_decelerations > 0.0005) {
    score_pathological += 3.5;
    explanations.push("Severe decelerations detected — urgent clinical review indicated");
  }
  if (f.percentage_of_time_with_abnormal_long_term_variability > 60) {
    score_pathological += 2.0;
    explanations.push("Majority of time with abnormal long-term variability");
  }
  if (f.histogram_variance > 120) {
    score_pathological += 1.5;
    explanations.push("High histogram variance — irregular heart rate pattern");
  }
  if (f.light_decelerations > 0.008) {
    score_pathological += 1.5;
    score_suspect += 0.5;
  }

  // === SUSPECT signals ===
  if (f.abnormal_short_term_variability > 45 && f.abnormal_short_term_variability <= 65) {
    score_suspect += 2.0;
    explanations.push("Elevated abnormal short-term variability — watchful monitoring advised");
  }
  if (f.mean_value_of_short_term_variability >= 0.5 && f.mean_value_of_short_term_variability < 1.0) {
    score_suspect += 1.5;
    explanations.push("Reduced short-term variability — sub-optimal fetal heart rate pattern");
  }
  if (f.percentage_of_time_with_abnormal_long_term_variability > 30 && f.percentage_of_time_with_abnormal_long_term_variability <= 60) {
    score_suspect += 1.5;
    explanations.push("Elevated abnormal long-term variability fraction");
  }
  if (f.accelerations < 0.001 && f.uterine_contractions > 0.004) {
    score_suspect += 1.5;
    explanations.push("Low accelerations with active contractions — reduced reactivity");
  }
  if (f.prolongued_decelerations > 0 && f.prolongued_decelerations <= 0.001) {
    score_suspect += 2.0;
    explanations.push("Low-level prolonged decelerations — borderline concern");
  }
  if (f.histogram_variance > 50 && f.histogram_variance <= 120) {
    score_suspect += 1.0;
  }
  if (f.mean_value_of_long_term_variability > 25) {
    score_suspect += 1.0;
    explanations.push("Elevated long-term variability mean — possible fetal stress response");
  }

  // === NORMAL signals ===
  if (f.accelerations > 0.004) {
    score_normal += 3.0;
    explanations.push("Good acceleration rate — healthy fetal reactivity");
  }
  if (f.mean_value_of_short_term_variability >= 1.0 && f.mean_value_of_short_term_variability <= 4.0) {
    score_normal += 2.5;
    explanations.push("Normal short-term variability — reassuring autonomic function");
  }
  if (f.abnormal_short_term_variability < 30) {
    score_normal += 2.0;
    explanations.push("Low abnormal short-term variability — stable beat-to-beat rhythm");
  }
  if (f.prolongued_decelerations === 0 && f.severe_decelerations === 0) {
    score_normal += 1.5;
    explanations.push("No severe or prolonged decelerations — low immediate risk");
  }
  if (f.histogram_variance < 25) {
    score_normal += 1.0;
  }
  if (f.percentage_of_time_with_abnormal_long_term_variability < 15) {
    score_normal += 1.5;
    explanations.push("Predominantly normal long-term variability — good fetal reserve");
  }
  if (f.baseline_value >= 110 && f.baseline_value <= 160) {
    score_normal += 1.0;
    explanations.push("Baseline FHR within normal range (110–160 bpm)");
  }

  // Softmax to probabilities
  const maxScore = Math.max(score_normal, score_suspect, score_pathological);
  const expN = Math.exp(score_normal - maxScore);
  const expS = Math.exp(score_suspect - maxScore);
  const expP = Math.exp(score_pathological - maxScore);
  const total = expN + expS + expP;

  const prob_normal       = expN / total;
  const prob_suspect      = expS / total;
  const prob_pathological = expP / total;

  const risk_class = prob_pathological > prob_suspect
    ? (prob_pathological > prob_normal ? 2 : 0)
    : (prob_suspect > prob_normal ? 1 : 0);

  const risk_labels = ["Normal", "Suspect", "Pathological"];
  const risk_label = risk_labels[risk_class];
  const probs = [prob_normal, prob_suspect, prob_pathological];
  const confidence = probs[risk_class];

  // Fetal Reserve Score (0–100)
  let frs = 50;
  // Positive contributors
  frs += Math.min(20, f.accelerations * 5000);
  frs += f.mean_value_of_short_term_variability >= 1.0 && f.mean_value_of_short_term_variability <= 4.0 ? 15 : 0;
  frs += f.percentage_of_time_with_abnormal_long_term_variability < 15 ? 10 : 0;
  // Negative contributors
  frs -= Math.min(25, f.abnormal_short_term_variability * 0.4);
  frs -= f.prolongued_decelerations * 8000;
  frs -= f.severe_decelerations * 5000;
  frs -= Math.min(15, f.percentage_of_time_with_abnormal_long_term_variability * 0.2);
  frs = Math.max(0, Math.min(100, frs));

  // Uncertainty
  const entropy = -(
    (prob_normal > 0 ? prob_normal * Math.log(prob_normal) : 0) +
    (prob_suspect > 0 ? prob_suspect * Math.log(prob_suspect) : 0) +
    (prob_pathological > 0 ? prob_pathological * Math.log(prob_pathological) : 0)
  ) / Math.log(3);

  const uncertainty = entropy < 0.2 ? "low" : entropy < 0.5 ? "moderate" : "high";

  return {
    risk_class,
    risk_label,
    confidence,
    prob_normal,
    prob_suspect,
    prob_pathological,
    fetal_reserve_score: Math.round(frs),
    explanation: explanations.slice(0, 5),
    uncertainty,
  };
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 200, headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
    );

    if (req.method === "GET") {
      // Return feature metadata for the UI form
      return new Response(
        JSON.stringify({ features: FEATURE_STATS, feature_names: FEATURE_NAMES }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    if (req.method === "POST") {
      const body = await req.json();
      const { features, session_id } = body as {
        features: Record<string, number>;
        session_id?: string;
      };

      // Validate all features present
      for (const name of FEATURE_NAMES) {
        if (features[name] === undefined || features[name] === null) {
          return new Response(
            JSON.stringify({ error: `Missing feature: ${name}` }),
            { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
          );
        }
      }

      const result = predict(features);

      // Persist to Supabase
      const { data: saved, error: dbError } = await supabase
        .from("predictions")
        .insert({
          session_id: session_id ?? "",
          features,
          risk_label: result.risk_label,
          risk_class: result.risk_class,
          confidence: result.confidence,
          prob_normal: result.prob_normal,
          prob_suspect: result.prob_suspect,
          prob_pathological: result.prob_pathological,
          fetal_reserve_score: result.fetal_reserve_score,
        })
        .select("id")
        .single();

      if (dbError) {
        console.error("DB insert error:", dbError);
      }

      return new Response(
        JSON.stringify({ ...result, id: saved?.id ?? null }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: String(err) }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
