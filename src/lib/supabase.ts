import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string | undefined
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_SUPABASE_ANON_KEY as string | undefined

export const SUPABASE_CONFIGURED = !!(supabaseUrl && supabaseAnonKey)

export const supabase = SUPABASE_CONFIGURED
  ? createClient(supabaseUrl!, supabaseAnonKey!)
  : null

export const EDGE_URL = SUPABASE_CONFIGURED
  ? `${supabaseUrl}/functions/v1/fetalyze-predict`
  : null

export const EDGE_HEADERS = SUPABASE_CONFIGURED
  ? {
      Authorization: `Bearer ${supabaseAnonKey}`,
      'Content-Type': 'application/json',
    }
  : null
