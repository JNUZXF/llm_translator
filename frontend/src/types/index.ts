export interface Language {
  code: string;
  name: string;
}

export interface TranslationScene {
  id: string;
  name: string;
  description: string;
}

export interface TranslationResponse {
  content?: string;
  done?: boolean;
  error?: string;
}

export interface PDFPage {
  page_number: number;
  text: string;
  char_count: number;
}

export interface PDFInfo {
  page_count: number;
  file_size: number;
  metadata: any;
}

export interface UploadResponse {
  success: boolean;
  filename: string;
  filepath: string;
  info: PDFInfo;
  pages: PDFPage[];
}

export interface FlowerType {
  type: string;
  colors: string[];
  petals: number;
  center: string;
}

export interface Flower {
  id: string;
  x: number;
  y: number;
  size: number;
  flowerType: FlowerType;
  speed: number;
  angle: number;
  rotation: number;
  opacity: number;
  bloomProgress: number;
  // 新增的物理属性
  vx: number;
  vy: number;
  targetAngle: number;
  wanderTimer: number;
  maxWanderTime: number;
}