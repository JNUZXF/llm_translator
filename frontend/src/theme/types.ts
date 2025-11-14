/**
 * 主题系统类型定义
 * 定义了完整的主题结构，确保类型安全和可扩展性
 */

// ============= 颜色类型 =============

/** 颜色值类型 */
export type ColorValue = string;

/** 调色板：包含主色和各种深浅变化 */
export interface Palette {
  main: ColorValue;
  light: ColorValue;
  lighter: ColorValue;
  dark: ColorValue;
  darker: ColorValue;
  contrast: ColorValue; // 对比色（用于文字等）
}

/** 语义化颜色 */
export interface SemanticColors {
  primary: Palette;
  secondary: Palette;
  success: Palette;
  warning: Palette;
  error: Palette;
  info: Palette;
}

/** 中性色系统 */
export interface NeutralColors {
  white: ColorValue;
  black: ColorValue;
  gray: {
    50: ColorValue;
    100: ColorValue;
    200: ColorValue;
    300: ColorValue;
    400: ColorValue;
    500: ColorValue;
    600: ColorValue;
    700: ColorValue;
    800: ColorValue;
    900: ColorValue;
  };
}

/** 背景色系统 */
export interface BackgroundColors {
  default: ColorValue;
  paper: ColorValue;
  elevated: ColorValue;
  overlay: ColorValue;
}

/** 文本色系统 */
export interface TextColors {
  primary: ColorValue;
  secondary: ColorValue;
  disabled: ColorValue;
  hint: ColorValue;
  inverse: ColorValue;
}

/** 边框色系统 */
export interface BorderColors {
  default: ColorValue;
  light: ColorValue;
  dark: ColorValue;
  focus: ColorValue;
}

// ============= 排版类型 =============

/** 字体家族 */
export interface FontFamily {
  primary: string;
  secondary: string;
  monospace: string;
}

/** 字体大小 */
export interface FontSize {
  xs: string;
  sm: string;
  base: string;
  lg: string;
  xl: string;
  '2xl': string;
  '3xl': string;
  '4xl': string;
  '5xl': string;
}

/** 字体粗细 */
export interface FontWeight {
  light: number;
  normal: number;
  medium: number;
  semibold: number;
  bold: number;
  black: number;
}

/** 行高 */
export interface LineHeight {
  none: number;
  tight: number;
  snug: number;
  normal: number;
  relaxed: number;
  loose: number;
}

/** 排版系统 */
export interface Typography {
  fontFamily: FontFamily;
  fontSize: FontSize;
  fontWeight: FontWeight;
  lineHeight: LineHeight;
  letterSpacing: {
    tighter: string;
    tight: string;
    normal: string;
    wide: string;
    wider: string;
    widest: string;
  };
}

// ============= 间距类型 =============

export interface Spacing {
  0: string;
  1: string;
  2: string;
  3: string;
  4: string;
  5: string;
  6: string;
  8: string;
  10: string;
  12: string;
  16: string;
  20: string;
  24: string;
  32: string;
  40: string;
  48: string;
  56: string;
  64: string;
}

// ============= 阴影类型 =============

export interface Shadows {
  none: string;
  xs: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  '2xl': string;
  inner: string;
}

// ============= 圆角类型 =============

export interface BorderRadius {
  none: string;
  xs: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  '2xl': string;
  '3xl': string;
  full: string;
}

// ============= 动画类型 =============

export interface Transitions {
  duration: {
    fast: string;
    normal: string;
    slow: string;
  };
  timing: {
    linear: string;
    easeIn: string;
    easeOut: string;
    easeInOut: string;
    spring: string;
  };
}

export interface Animations {
  fadeIn: string;
  fadeOut: string;
  slideUp: string;
  slideDown: string;
  slideLeft: string;
  slideRight: string;
  scaleUp: string;
  scaleDown: string;
  rotate: string;
  pulse: string;
  bounce: string;
}

// ============= Z-Index类型 =============

export interface ZIndex {
  base: number;
  dropdown: number;
  sticky: number;
  fixed: number;
  modalBackdrop: number;
  modal: number;
  popover: number;
  tooltip: number;
}

// ============= 断点类型 =============

export interface Breakpoints {
  xs: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  '2xl': string;
}

// ============= 主题配置类型 =============

/** 主题模式 */
export type ThemeMode = 'light' | 'dark';

/** 完整主题配置 */
export interface Theme {
  mode: ThemeMode;
  name: string;
  colors: {
    semantic: SemanticColors;
    neutral: NeutralColors;
    background: BackgroundColors;
    text: TextColors;
    border: BorderColors;
  };
  typography: Typography;
  spacing: Spacing;
  shadows: Shadows;
  borderRadius: BorderRadius;
  transitions: Transitions;
  animations: Animations;
  zIndex: ZIndex;
  breakpoints: Breakpoints;
}

// ============= 主题上下文类型 =============

export interface ThemeContextType {
  theme: Theme;
  mode: ThemeMode;
  setMode: (mode: ThemeMode) => void;
  toggleMode: () => void;
  availableThemes: string[];
  currentTheme: string;
  setTheme: (themeName: string) => void;
}

// ============= 主题配置选项 =============

export interface ThemeOptions {
  /** 默认主题模式 */
  defaultMode?: ThemeMode;
  /** 是否持久化主题设置到localStorage */
  persist?: boolean;
  /** localStorage的key */
  storageKey?: string;
  /** 自定义主题 */
  customThemes?: Record<string, Partial<Theme>>;
}
