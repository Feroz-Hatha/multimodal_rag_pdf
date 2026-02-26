export interface SessionDoc {
  document_id: string;
  filename: string;
  total_chunks: number;
  text_chunks: number;
  table_chunks: number;
  image_chunks: number;
}

export interface Source {
  filename: string;
  heading: string;
  section_hierarchy: string[];
  page_numbers: number[];
  content_type: string;
  score: number;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'running' | 'done' | 'error';
  progress: number;
  stage: string;
  filename: string;
  document_id?: string;
  total_chunks?: number;
  text_chunks?: number;
  table_chunks?: number;
  image_chunks?: number;
  already_indexed?: boolean;
  error?: string;
}

export interface QueryResponse {
  question: string;
  answer: string;
  sources: Source[];
  model_id: string;
  input_tokens: number;
  output_tokens: number;
  estimated_cost_usd: number;
}
