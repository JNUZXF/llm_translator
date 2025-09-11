export const API_BASE_URL = 'http://localhost:5000/api';

export const SUPPORTED_LANGUAGES = [
  { code: 'de', name: 'Deutsch' },
  { code: 'fr', name: 'Français' },
  { code: 'zh', name: '中文' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Español' },
  { code: 'it', name: 'Italiano' },
  { code: 'pt', name: 'Português' },
  { code: 'ru', name: 'Русский' },
  { code: 'ja', name: '日本語' },
  { code: 'ko', name: '한국어' },
  { code: 'ar', name: 'العربية' },
  { code: 'hi', name: 'हिन्दी' },
  { code: 'th', name: 'ไทย' },
  { code: 'vi', name: 'Tiếng Việt' },
  { code: 'nl', name: 'Nederlands' },
  { code: 'sv', name: 'Svenska' },
  { code: 'da', name: 'Dansk' },
  { code: 'no', name: 'Norsk' },
  { code: 'fi', name: 'Suomi' },
  { code: 'pl', name: 'Polski' },
  { code: 'tr', name: 'Türkçe' },
  { code: 'he', name: 'עברית' },
];

// 真实花朵颜色配置
export const FLOWER_TYPES = [
  {
    type: 'rose',
    colors: ['#FF6B9D', '#FF8FB1', '#FFB3C6'],
    petals: 6,
    center: '#FFD93D'
  },
  {
    type: 'cherry',
    colors: ['#FFB3C6', '#FFC0CB', '#FFCCCB'],
    petals: 5,
    center: '#FF69B4'
  },
  {
    type: 'lavender',
    colors: ['#E6E6FA', '#DDA0DD', '#DA70D6'],
    petals: 4,
    center: '#9370DB'
  },
  {
    type: 'sunflower',
    colors: ['#FFD700', '#FFA500', '#FF8C00'],
    petals: 12,
    center: '#8B4513'
  },
  {
    type: 'lotus',
    colors: ['#F0E68C', '#FFFAF0', '#FFF8DC'],
    petals: 8,
    center: '#FFB347'
  },
  {
    type: 'hibiscus',
    colors: ['#FF1493', '#FF69B4', '#FFB6C1'],
    petals: 5,
    center: '#DC143C'
  },
  {
    type: 'peony',
    colors: ['#FFE5E5', '#FFCCCB', '#FFB6C1'],
    petals: 10,
    center: '#FF69B4'
  },
  {
    type: 'daisy',
    colors: ['#FFFFFF', '#FFFAF0', '#F5F5DC'],
    petals: 12,
    center: '#FFD700'
  },
  {
    type: 'tulip',
    colors: ['#FF4500', '#FF6347', '#FF7F50'],
    petals: 6,
    center: '#FFA500'
  },
  {
    type: 'iris',
    colors: ['#9370DB', '#8A2BE2', '#9932CC'],
    petals: 6,
    center: '#4B0082'
  },
  {
    type: 'orchid',
    colors: ['#DA70D6', '#DDA0DD', '#EE82EE'],
    petals: 5,
    center: '#8B008B'
  },
  {
    type: 'magnolia',
    colors: ['#FFF8DC', '#FFFACD', '#FFEFD5'],
    petals: 9,
    center: '#DEB887'
  }
];

export const FLOWER_COLORS = [
  '#FF6B9D', '#FFB3C6', '#E6E6FA', '#FFD700', '#F0E68C', '#FF1493'
];